from __future__ import unicode_literals
from contextlib import contextmanager
import getpass
from io import BytesIO
import json
import os
import re
import rlcompleter
import socket
import struct
import sys
import time
try:
    from base64 import encodebytes
except ImportError:
    from base64 import encodestring as encodebytes

class BKserver():
    config_dir = os.path.join(os.environ["HOME"], ".config", "bridge_kernel")
    localhost = socket.gethostname()
    frame_format = "!I"
    frame_length = struct.calcsize(frame_format)
    max_message_length = (2**(8 * frame_length)) - 1
    chunk_size = 1024

    class ClientFile():
        def __init__(self, client, id_str):
            self.client = client
            self.id_str = id_str
        def write(self, s):
            self.client.writemsg(dict(type=self.id_str, code=s))
        def flush(self):
            pass

    class NullFile():
        def write(self, s):
            pass
        def flush(self):
            pass

    def __init__(self, comm, ns, prefix=None):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nprocs = comm.Get_size()
        self.exec_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.ns = ns
        self.files = []
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log = sys.__stdout__.write
        self.go = True
        if self.rank == 0:
            self.client_stdout = self.ClientFile(self, "stdout")
            self.client_stderr = self.ClientFile(self, "stderr")
        else:
            self.client_stdout = self.NullFile()
            self.client_stderr = self.NullFile()
            
        if prefix is not None:
            self.prefix = prefix
        else:
            self.prefix = "%s_%s_%s" % (self.exec_name, self.localhost,
                                        re.sub(r"\W", "_", self.start_time))

        self.config_file = os.path.join(self.config_dir,
                                        "{}.json".format(self.prefix))
        self.addr = os.path.join(self.config_dir,
                                 "{}-ipc".format(self.prefix))

        if self.rank == 0:
            if not os.path.exists(self.config_dir):
                os.makedirs(self.config_dir)

        #ns["exit"] = self._quitter


    def _quitter(self):
        self.go = False

    @contextmanager
    def _redirect(self):
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = self.client_stdout
        sys.stderr = self.client_stderr
        try:
            yield
        finally:
            sys.stdout = stdout
            sys.stderr = stderr

    def readmsg(self):
        try:
            header = self._client_sock.recv(self.frame_length)
        except socket.error:
            print("client disconnected")
            self._client_sock = None
            return None
        if header is None or len(header) == 0:
            print("client disconnected")
            self._client_sock = None
            return None

        length = struct.unpack(self.frame_format, header)[0]
        bytes_read = 0
        chunks = []
        if length > 0:
            while bytes_read < length:
                to_read = min(self.chunk_size, length - bytes_read)
                chunk = self._client_sock.recv(to_read)
                if chunk is None:
                    return
                chunks.append(chunk)
                bytes_read += len(chunk)
            message = b"".join(chunks)
        else:
            message = None

        try:
            obj = json.loads(message.decode("utf-8"))
        except ValueError:
            print("invalid message format")
            return None

        return obj

    def writemsg(self, data):
        serial = json.dumps(data, ensure_ascii=False)
        header = struct.pack(self.frame_format, len(serial))
        message = header + serial.encode("utf-8")
        message_length = len(message)

        if message_length > self.max_message_length:
            self.log("fatal: message too long\n")
            return

        bytes_sent = 0
        while bytes_sent < message_length:
            try:
                chunk = message[bytes_sent:bytes_sent
                                + min(self.chunk_size,
                                      message_length - bytes_sent)]
                sent = self._client_sock.send(chunk)
            except socket.error as sockerr:
                (code, msg) = sockerr.args
                self.debug("socket error: {}, {}".format(code, msg))
                return

            if sent == 0:
                self.post_message("error: socket connection broken")
                return
            bytes_sent += sent

    def execute(self, code):
        with self._redirect():
            if self.rank == 0:
                self.log("py> {}\n".format(code))
            try:
                resp = eval(code, self.ns)
                if resp is not None:
                    resp = str(resp)
            except Exception as e:
                try:
                    exec(code, self.ns)
                    resp = None
                except Exception as e:
                    resp = str(e)
            if self.rank == 0 and resp is not None and len(resp) > 0:
                try:
                    resp = resp + "\n"
                    self.log("{}".format(resp))
                except UnicodeDecodeError:
                    self.log("<cannot display, non-ascii>\n")
                self.writemsg({"type": "stdout", "code": resp})
                if resp.startswith('STOP'):
                    self._quitter()

    def complete(self, text, cursor_pos=None):
        if self.rank == 0:
            if cursor_pos is None:
                cursor_pos = len(text)
            cursor_start = 0
            for i in range(cursor_pos - 1, -1, -1):
                if text[i].isalpha() or text[i].isnumeric() \
                   or text[i] == "_" or text[i] == ".":
                    pass
                else:
                    cursor_start = i + 1
                    break

            text = text[cursor_start:cursor_pos]
            completor = rlcompleter.Completer(namespace=self.ns)

            if len(text) == 0:
                matches = completor.global_matches("")
            else:
                completor.complete(text, 0)
                if hasattr(completor, "matches"):
                    matches = completor.matches
                else:
                    matches = None

            self.writemsg({
                "type": "complete",
                "cursor_start": cursor_start,
                "cursor_end": cursor_pos,
                "matches": [m[0:-1] if m[-1] == "(" else m for m in matches]
            })

    def _remove_files(self):
        if self.files is not None:
            while len(self.files) > 0:
                f = self.files.pop()
                if os.path.exists(f):
                    print("removing {}".format(f))
                    os.remove(f)



    def Serve(self):
        if self.rank == 0:
            # create socket, bind and listen, and create json
            try:
                os.unlink(self.addr)
            except OSError:
                if os.path.exists(self.addr):
                    raise
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            print("trying to bind to {}... ".format(self.addr))
            sock.bind(self.addr)
            self.files.append(self.addr)

            hosts = []
            hosts.append(self.localhost)

            config = {
                "protocol": "ipc",
                "uds": self.addr,
                "hosts": hosts,
                "code": self.exec_name,
                "argv": " ".join(sys.argv),
                "user": getpass.getuser(),
                "date": self.start_time
            }

            if config is None or sock is None:
                print("fatal: failed to create socket")
                return

            print("connect to: {}".format(self.config_file))
            with open(self.config_file, "w") as _:
                json.dump(config, _, indent=4, sort_keys=True)
            self.files.append(self.config_file)

            sock.listen(0)

            self.config = config
            self._sock = sock

            print("bind successful. waiting for client...")
            self._client_sock, addr = sock.accept()
            print("got client - {}".format(addr))

            self.writemsg({"type": "idle"})

        #---------------- event loop ----------------
        while self.go:
            data = self.readmsg()
            if data is None:
                print("disconnected")
                break

            if data["type"] == "execute":
                self.execute(data["code"])
                if not self.go: break
            elif data["type"] == "complete":
                self.complete(data["code"], cursor_pos=data["cursor_pos"])
            elif data["type"] == "disconnect":
                break # goto epilogue

            self.writemsg({"type": "idle"})
            continue # end of while(self.go)

        #---------------- epilogue ----------------
        eval("stop()", self.ns)
        self._quitter()
        self._remove_files()
        
