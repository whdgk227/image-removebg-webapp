import http.server
import socketserver
import sys

PORT = 9090


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def handle_error(self, request, client_address):
        exc_type = sys.exc_info()[0]
        if exc_type in (ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)

    def log_message(self, format, *args):
        super().log_message(format, *args)


class ReusableServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc_type = sys.exc_info()[0]
        if exc_type in (ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)


with ReusableServer(("", PORT), QuietHandler) as httpd:
    print(f"[INFO] Static server running on port {PORT}")
    httpd.serve_forever()
