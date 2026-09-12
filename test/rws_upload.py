import argparse
import html
import os
import re
import sys

import requests
import urllib3
from requests.auth import HTTPDigestAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}


def post_form(session, url, data=None):
    return session.post(url, data=(data or {}), headers=FORM_HEADERS)

TASK = "T_ROB_R"
REMOTE_DIR = "$HOME"
KNOWN_MODULE_NAMES = {"EGMTest", "TCPTestR", "TCPTestL", "TrajExecR", "TrajExecL"}

CREDENTIAL_CANDIDATES = [
    ("Admin", "robotics"),
    ("Default User", "robotics"),
]


def find_working_credentials(base, session):
    for user, password in CREDENTIAL_CANDIDATES:
        session.auth = HTTPDigestAuth(user, password)
        r = session.get(f"{base}/", timeout=10)
        print(f"   trying {user!r}/{password!r} -> HTTP {r.status_code}")
        if r.status_code == 200:
            return user, password
    return None, None


def get_module_name(content: bytes) -> str:
    m = re.search(r"^\s*MODULE\s+(\w+)", content.decode(errors="replace"), re.MULTILINE)
    if not m:
        raise SystemExit("Couldn't find a 'MODULE <name>' declaration in this file")
    return m.group(1)


def list_loaded_program_modules(session, base, task):
    r = session.get(f"{base}/rw/rapid/modules", params={"task": task})
    if r.status_code != 200:
        print(f"   (couldn't list existing modules: HTTP {r.status_code}, skipping cleanup)")
        return []
    names = re.findall(r'<span class="name">([^<]+)</span>', r.text)
    types = re.findall(r'<span class="type">([^<]+)</span>', r.text)
    return [n for n, t in zip(names, types) if t == "ProgMod"]


def get_module_text(session, base, task, module_name, max_chars=4000):
    r = session.get(f"{base}/rw/rapid/modules/{module_name}",
                     params={"resource": "module-text", "task": task})
    if r.status_code != 200:
        return f"(couldn't fetch module text: HTTP {r.status_code})\n{r.text[:1000]}"

    # RWS sometimes inlines the source directly in a <span class="module-text">;
    # other times (seemingly for longer modules) it writes it to a temp file on
    # the controller instead and reports that file's path back to us, requiring
    # a second fileservice request to actually read it.
    m = re.search(r'<span class="module-text">(.*?)</span>', r.text, re.DOTALL)
    if m:
        text = html.unescape(m.group(1)).strip("\n")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more characters]"
        return text

    m = re.search(r'<span class="file-path">"?([^"<]+?)"?</span>', r.text)
    if not m:
        text = r.text
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more characters]"
        return "(couldn't find module-text or a file-path in the response; showing raw response)\n" + text

    file_path = m.group(1).strip().strip('"')
    fs_path = file_path.lstrip("/")
    file_url = f"{base}/fileservice/{fs_path}"
    fr = session.get(file_url)
    if fr.status_code != 200:
        return (f"(module-text pointed at {file_path!r} but GET {file_url} -> "
                 f"HTTP {fr.status_code})\n{fr.text[:1000]}")

    text = fr.text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more characters]"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="path to the .modx file to upload, e.g. TCPTest_ROB_R.modx")
    ap.add_argument("--task", default=TASK, help=f"RAPID task to load into (default: {TASK})")
    ap.add_argument("--host", default="192.168.125.1", help="controller IP (default: 192.168.125.1)")
    ap.add_argument("--user", default=None, help="skip auto-detection, force this username")
    ap.add_argument("--password", default="robotics")
    ap.add_argument("--https", action="store_true", help="use https instead of http")
    args = ap.parse_args()

    module_path = args.module
    module_filename = os.path.basename(module_path).replace(".modx", ".mod")
    task = args.task

    scheme = "https" if args.https else "http"
    base = f"{scheme}://{args.host}"
    session = requests.Session()
    session.verify = False

    if args.user:
        user, password = args.user, args.password
        print(f"Using forced credentials: {user!r}")
    else:
        print("0) Detecting RWS credentials ...")
        user, password = find_working_credentials(base, session)
        if user is None:
            print("\n   Neither 'Admin' nor 'Default User' (password 'robotics') worked.")
            print("   Check the controller is reachable (ping) and that RWS credentials")
            print("   haven't been customized further. Stopping here -- nothing was")
            print("   uploaded or changed on the controller.")
            sys.exit(1)
        print(f"   Using {user!r} / {password!r}")

    session.auth = HTTPDigestAuth(user, password)

    with open(module_path, "rb") as f:
        content = f.read()

    print(f"\n1) Uploading {module_filename} to {REMOTE_DIR} ...")
    upload_url = f"{base}/fileservice/{REMOTE_DIR}/{module_filename}"
    r = session.put(upload_url, data=content, headers={"Content-Type": "text/plain"})
    print(f"   PUT {upload_url} -> {r.status_code}")
    if r.status_code not in (200, 201, 204):
        print("   Upload failed. Response body:")
        print(" ", r.text[:2000])
        sys.exit(1)

    print("\n2) Requesting RAPID mastership ...")
    mastership_url = f"{base}/rw/mastership?action=request"
    r = post_form(session, mastership_url)
    print(f"   POST {mastership_url} -> {r.status_code}")
    if r.status_code not in (200, 201, 204):
        print("   Mastership request failed (another client may already hold it, or the")
        print("   controller is in a mode that doesn't allow it). Response body:")
        print(" ", r.text[:2000])
        sys.exit(1)

    new_module_name = get_module_name(content)
    print(f"\n3) Checking for other program modules already in {task} "
          f"(new module is {new_module_name!r}) ...")
    existing = list_loaded_program_modules(session, base, task)
    others = [n for n in existing if n != new_module_name]
    known = [n for n in others if n in KNOWN_MODULE_NAMES]
    unrecognized = [n for n in others if n not in KNOWN_MODULE_NAMES]

    to_unload = list(known)
    if unrecognized:
        print(f"   Found module(s) this script doesn't recognize as ours:")
        for i, name in enumerate(unrecognized, 1):
            print(f"     {i}. {name}")
        print()
        for i, name in enumerate(unrecognized, 1):
            print(f"   --- {i}. {name} " + "-" * max(1, 40 - len(name)))
            print(get_module_text(session, base, task, name))
            print()
        print("   These may belong to someone else -- nothing is removed unless you say so")
        print("   here, one at a time. Leave blank to remove none of them.")
        selection = input("   Enter number(s) to unload, space or comma separated (or blank): ").strip()
        chosen = []
        if selection:
            for tok in re.split(r"[,\s]+", selection):
                tok = tok.strip()
                if not tok:
                    continue
                if not tok.isdigit() or not (1 <= int(tok) <= len(unrecognized)):
                    print(f"   Ignoring invalid selection {tok!r}.")
                    continue
                chosen.append(unrecognized[int(tok) - 1])
        left_alone = [n for n in unrecognized if n not in chosen]
        if chosen:
            print(f"   Will unload (explicitly chosen): {chosen}")
        if left_alone:
            print(f"   Leaving loaded (not selected): {left_alone}")
            print("   If any of these define a routine with the same name used by the new")
            print("   module (e.g. another 'main'), the load below may still fail.")
        to_unload += chosen

    for name in to_unload:
        print(f"   Unloading {name!r} ...")
        r = post_form(session, f"{base}/rw/rapid/tasks/{task}?action=unloadmod", {"module": name})
        print(f"   POST .../tasks/{task}?action=unloadmod (module={name}) -> {r.status_code}")
        if r.status_code not in (200, 201, 204):
            print("   Unload failed. Response body:")
            print(" ", r.text[:2000])
            post_form(session, f"{base}/rw/mastership?action=release")
            sys.exit(1)
    if not others:
        print("   Nothing else loaded.")

    print(f"\n4) Loading module into task {task} ...")
    load_url = f"{base}/rw/rapid/tasks/{task}?action=loadmod"
    remote_module_path = f"{REMOTE_DIR}/{module_filename}"
    r = post_form(session, load_url, {"modulepath": remote_module_path, "replace": "true"})
    print(f"   POST {load_url} -> {r.status_code}")
    if r.status_code not in (200, 201, 204):
        print("   Load failed. Response body:")
        print(" ", r.text[:2000])
        post_form(session, f"{base}/rw/mastership?action=release")
        sys.exit(1)

    print("\n5) Resetting program pointer to main ...")
    resetpp_url = f"{base}/rw/rapid/execution?action=resetpp"
    r = post_form(session, resetpp_url)
    print(f"   POST {resetpp_url} -> {r.status_code}")
    if r.status_code not in (200, 201, 204):
        print("   resetpp failed. Response body:")
        print(" ", r.text[:2000])
        post_form(session, f"{base}/rw/mastership?action=release")
        sys.exit(1)

    print("\n6) Releasing mastership ...")
    r = post_form(session, f"{base}/rw/mastership?action=release")
    print(f"   POST {base}/rw/mastership?action=release -> {r.status_code}")

    print("\n=== DONE ===")
    print(f"Module uploaded and loaded, PP is at main on {task}.")


if __name__ == "__main__":
    main()
