import argparse
import socket
import threading
import time

EGM_PORT = 6511
TCP_PORT_RIGHT = 6512
TCP_PORT_LEFT = 6513
ROBOT_IP = "192.168.125.1"

_print_lock = threading.Lock()


def _tcp_connect(port, label, module_name):
    print(f"[{label}] Connecting to robot at {ROBOT_IP}:{port}. Make sure the task running "
          f"{module_name} is already Played and waiting.")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ROBOT_IP, port))
    print(f"[{label}] Connected to robot.")
    return sock


def _tcp_recv_joints(sock, label):
    data = sock.recv(4096).decode().strip()
    joints = [float(v) for v in data.split(",")]
    with _print_lock:
        _print_joints(joints, label)
    return joints


def tcp_listen(port=TCP_PORT_RIGHT, label="tcp", module_name="TCPTestR"):
    sock = _tcp_connect(port, label, module_name)
    _tcp_recv_joints(sock, label)
    sock.sendall(b"STOP")
    print(f"\n[{label}] Sent STOP -- no motion requested. Connection + reporting confirmed working.")
    sock.close()


def tcp_listen_both():
    threads = [
        threading.Thread(target=tcp_listen, args=(TCP_PORT_RIGHT, "tcp-R", "TCPTestR")),
        threading.Thread(target=tcp_listen, args=(TCP_PORT_LEFT, "tcp-L", "TCPTestL")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\nBoth arms connected, reported, and closed successfully -- simultaneous "
          "dual-arm TCP confirmed working.")


def tcp_nudge(joint_index, degrees, port=TCP_PORT_RIGHT, label="tcp", module_name="TCPTestR"):
    if not (1 <= joint_index <= 7):
        raise SystemExit("--joint must be 1-7 for --protocol tcp")

    sock = _tcp_connect(port, label, module_name)
    joints = _tcp_recv_joints(sock, label)

    current = joints[joint_index - 1]
    target = current + degrees
    print(f"\n[{label}] Will request axis {joint_index}: {current:.3f} -> {target:.3f} deg "
          f"({degrees:+.3f} deg). RAPID hard-clamps this to +-3 deg regardless.")
    input(f"[{label}] Press Enter to send the command (Ctrl-C to abort) ...")

    cmd = f"NUDGE,{joint_index},{degrees}"
    sock.sendall(cmd.encode())
    print(f"[{label}] Sent: {cmd}")

    resp = sock.recv(4096).decode().strip()
    print(f"[{label}] Received: {resp}")
    if resp.startswith("DONE,"):
        with _print_lock:
            _print_joints([float(v) for v in resp[len("DONE,"):].split(",")], label)

    sock.close()


def _print_joints(joints, label="tcp"):
    names = [f"axis{i+1}" for i in range(len(joints))]
    print(f"[{label}] Joints (deg):")
    for n, v in zip(names, joints):
        print(f"  [{label}] {n} = {v:.3f}")


def _egm_recv(sock, egm_pb2, timeout=5.0):
    sock.settimeout(timeout)
    data, addr = sock.recvfrom(65536)
    msg = egm_pb2.EgmRobot()
    msg.ParseFromString(data)
    return msg, addr


def _egm_print_feedback(msg):
    j = list(msg.feedBack.joints.joints) if msg.feedBack.HasField("joints") else []
    motors = msg.motorState.state if msg.HasField("motorState") else None
    rapid = msg.rapidExecState.state if msg.HasField("rapidExecState") else None
    print(f"  seqno={msg.header.seqno:6d}  joints(deg, axes 1-6)={['%.3f' % v for v in j]}  "
          f"motors={motors}  rapid_state={rapid}")


def egm_listen(n_messages):
    egm_pb2 = _import_egm_pb2()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", EGM_PORT))
    try:
        print(f"[egm] Listening on UDP port {EGM_PORT} for the robot's EGM session...")
        print("[egm] (Start/run the EGM RAPID routine on the controller now if it isn't already.)")
        last_addr = None
        for _ in range(n_messages):
            msg, addr = _egm_recv(sock, egm_pb2, timeout=10.0)
            last_addr = addr
            _egm_print_feedback(msg)
        print(f"\n[egm] Received {n_messages} feedback messages from {last_addr}. Session is alive.")
    finally:
        sock.close()


def egm_nudge(joint_index, degrees, hold_seconds):
    if not (1 <= joint_index <= 6):
        raise SystemExit("--joint must be 1-6 for --protocol egm (axis 7 isn't wired up over EGM yet)")

    egm_pb2 = _import_egm_pb2()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", EGM_PORT))
    try:
        print(f"[egm] Listening on UDP port {EGM_PORT} for the robot's EGM session...")
        msg, addr = _egm_recv(sock, egm_pb2, timeout=10.0)
        print("[egm] First feedback received:")
        _egm_print_feedback(msg)

        base_joints = list(msg.feedBack.joints.joints)
        idx0 = joint_index - 1
        if idx0 >= len(base_joints):
            raise SystemExit(f"joint index {joint_index} out of range for {len(base_joints)} joints")

        target = list(base_joints)
        target[idx0] += degrees
        print(f"\n[egm] Will nudge axis {joint_index} from {base_joints[idx0]:.3f} deg "
              f"to {target[idx0]:.3f} deg ({degrees:+.3f} deg) and hold for {hold_seconds}s.")
        input("Press Enter to send the command (Ctrl-C to abort) ...")

        seqno = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < hold_seconds:
            sock.settimeout(2.0)
            try:
                data, _ = sock.recvfrom(65536)
                fb = egm_pb2.EgmRobot()
                fb.ParseFromString(data)
            except socket.timeout:
                break

            cmd = egm_pb2.EgmSensor()
            cmd.header.seqno = seqno
            cmd.header.mtype = egm_pb2.EgmHeader.MSGTYPE_CORRECTION
            cmd.planned.joints.joints.extend(target)
            sock.sendto(cmd.SerializeToString(), addr)
            seqno += 1
            if seqno % 20 == 0:
                print(f"  sent {seqno} correction messages, current feedback:")
                _egm_print_feedback(fb)

        print(f"\n[egm] Done. Sent {seqno} correction messages targeting axis {joint_index} "
              f"= {target[idx0]:.3f} deg.")
    finally:
        sock.close()


def _import_egm_pb2():
    import os
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") != "python":
        raise SystemExit(
            "Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python before running --protocol egm "
            "(the installed protobuf runtime is newer than what protoc generated egm_pb2.py for).\n"
            "  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 robot_test.py --protocol egm ..."
        )
    import egm_pb2
    return egm_pb2


def main():
    parser = argparse.ArgumentParser(description="Unified EGM/TCP smoke test for YuMi")
    parser.add_argument("--protocol", choices=["tcp", "egm"], required=True)
    parser.add_argument("--mode", choices=["listen", "nudge"], default="listen")
    parser.add_argument("--arm", choices=["right", "left", "both"], default="right",
                         help="tcp only; egm is right-arm-only for now. 'both' is listen-mode only.")
    parser.add_argument("--joint", type=int, default=1,
                         help="1-indexed physical axis number (1-7 for tcp, 1-6 for egm)")
    parser.add_argument("--degrees", type=float, default=None,
                         help="nudge mode: how far to move it (default 2 for tcp, 0.5 for egm)")
    parser.add_argument("--messages", type=int, default=10,
                         help="egm listen mode: how many feedback messages to print")
    parser.add_argument("--hold-seconds", type=float, default=5.0,
                         help="egm nudge mode: how long to hold the target")
    args = parser.parse_args()

    if args.degrees is None:
        args.degrees = 2.0 if args.protocol == "tcp" else 0.5

    if args.protocol == "tcp":
        arm_config = {
            "right": (TCP_PORT_RIGHT, "tcp-R", "TCPTestR"),
            "left": (TCP_PORT_LEFT, "tcp-L", "TCPTestL"),
        }
        if args.mode == "listen":
            if args.arm == "both":
                tcp_listen_both()
            else:
                port, label, module_name = arm_config[args.arm]
                tcp_listen(port, label, module_name)
        else:
            if args.arm == "both":
                raise SystemExit("--arm both is listen-mode only for now; nudge one arm at a time.")
            port, label, module_name = arm_config[args.arm]
            tcp_nudge(args.joint, args.degrees, port, label, module_name)
    else:
        if args.mode == "listen":
            egm_listen(args.messages)
        else:
            egm_nudge(args.joint, args.degrees, args.hold_seconds)


if __name__ == "__main__":
    main()
