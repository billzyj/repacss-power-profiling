"""SSH tunnel helper for reaching the private eGauge network."""

from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass

from shared.errors import EGaugeError


@dataclass
class SSHLocalForward:
    """Manage a subprocess-based SSH local port forward."""

    ssh_host: str
    ssh_port: int
    ssh_user: str
    target_host: str
    target_port: int
    keepalive_interval: int
    local_bind_host: str = "127.0.0.1"
    private_key_path: str | None = None

    def __post_init__(self) -> None:
        self.local_port: int | None = None
        self.process: subprocess.Popen | None = None

    def start(self) -> int:
        """Start the SSH port forward and return the allocated local port."""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.local_bind_host, 0))
            self.local_port = sock.getsockname()[1]

        ssh_cmd = [
            "ssh",
            "-N",
            "-L",
            f"{self.local_bind_host}:{self.local_port}:{self.target_host}:{self.target_port}",
            "-p",
            str(self.ssh_port),
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ServerAliveInterval={self.keepalive_interval}",
        ]
        if self.private_key_path:
            ssh_cmd += ["-o", "IdentitiesOnly=yes", "-i", self.private_key_path]

        destination = f"{self.ssh_user}@{self.ssh_host}" if self.ssh_user else self.ssh_host
        ssh_cmd.append(destination)

        self.process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)

        if self.process.poll() is not None:
            _, stderr = self.process.communicate()
            raise EGaugeError(f"SSH tunnel failed: {stderr.decode().strip()}")

        return self.local_port

    def stop(self) -> None:
        """Stop the SSH port forward if it is running."""

        if not self.process:
            return
        self.process.terminate()
        self.process.wait(timeout=5)
        self.process = None
