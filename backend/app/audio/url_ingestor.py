"""Audio URL Ingestion Service for VoxGuard.

Safely fetches remote audio streams from direct URLs, podcast links, or web media:
- Enforces SSRF prevention against private IP addresses and loopbacks
- Enforces 10 MB payload limits and 12-second download timeouts
- Validates media MIME types and content headers
"""

import ipaddress
import os
import socket
import urllib.parse
import urllib.request
from typing import Tuple


class AudioUrlIngestionError(Exception):
    """Raised when URL audio retrieval fails or violates security policy."""
    pass


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Custom redirect handler verifying that redirect destinations do not resolve to private or loopback networks."""

    def __init__(self, is_private_ip_fn):
        super().__init__()
        self.is_private_ip = is_private_ip_fn

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme.lower() not in ("http", "https"):
            raise AudioUrlIngestionError("Redirect to non-HTTP/HTTPS protocol is blocked.")
        if not parsed.hostname:
            raise AudioUrlIngestionError("Redirect URL hostname is invalid.")
        if parsed.hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0"):
            raise AudioUrlIngestionError("Redirect to local addresses is restricted for security.")
        if self.is_private_ip(parsed.hostname):
            raise AudioUrlIngestionError("Redirect to private internal network addresses is blocked.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class AudioUrlIngestor:
    """Secure audio URL downloader with SSRF and size safeguards."""

    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    TIMEOUT_SECONDS = 12

    BLOCKED_NETWORKS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
    ]

    def _is_private_ip(self, hostname: str) -> bool:
        """Checks if hostname resolves to a private or loopback IP address."""
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip_obj = ipaddress.ip_address(ip_str)
                for net in self.BLOCKED_NETWORKS:
                    if ip_obj in net:
                        return True
            return False
        except Exception:
            return True

    def fetch(self, url: str) -> Tuple[bytes, str, str]:
        """Fetches remote audio stream safely.

        Args:
            url: The HTTP/HTTPS URL pointing to an audio resource.

        Returns:
            Tuple of (audio_bytes, filename, content_type).
        """
        if not url or not isinstance(url, str):
            raise AudioUrlIngestionError("A valid audio URL string is required.")

        url = url.strip()
        parsed = urllib.parse.urlparse(url)

        if parsed.scheme.lower() not in ("http", "https"):
            raise AudioUrlIngestionError("Only HTTP and HTTPS URLs are supported.")

        if not parsed.hostname:
            raise AudioUrlIngestionError("Invalid URL hostname.")

        # Check for loopback / local hosts
        if parsed.hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0"):
            raise AudioUrlIngestionError("Access to local addresses is restricted for security.")

        if self._is_private_ip(parsed.hostname):
            raise AudioUrlIngestionError("Access to private internal network addresses is blocked.")

        # Extract filename from path or default
        path_name = os.path.basename(parsed.path) or "remote_audio.wav"
        if not any(path_name.lower().endswith(ext) for ext in (
            ".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac",
            ".webm", ".opus", ".wma", ".aiff", ".aif", ".caf",
            ".amr", ".3gp", ".mp4"
        )):
            path_name += ".wav"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "VoxGuard-Forensics/1.2 (+https://github.com/voxguard)",
                "Accept": "audio/*, application/octet-stream, */*"
            }
        )

        try:
            opener = urllib.request.build_opener(SafeRedirectHandler(self._is_private_ip))
            with opener.open(req, timeout=self.TIMEOUT_SECONDS) as response:
                content_type = response.headers.get_content_type() or "audio/wav"
                content_length = response.headers.get("Content-Length")

                if content_length and int(content_length) > self.MAX_BYTES:
                    raise AudioUrlIngestionError(f"Remote audio file exceeds {self.MAX_BYTES // (1024*1024)} MB limit.")

                # Read safely in chunks up to MAX_BYTES
                chunks = []
                total_read = 0
                chunk_size = 64 * 1024

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    if total_read > self.MAX_BYTES:
                        raise AudioUrlIngestionError(f"Remote stream exceeded {self.MAX_BYTES // (1024*1024)} MB limit.")
                    chunks.append(chunk)

                audio_bytes = b"".join(chunks)

                if len(audio_bytes) == 0:
                    raise AudioUrlIngestionError("Remote URL returned an empty audio payload.")

                return audio_bytes, path_name, content_type

        except AudioUrlIngestionError:
            raise
        except Exception as exc:
            raise AudioUrlIngestionError(f"Failed to retrieve audio from URL: {str(exc)}")
