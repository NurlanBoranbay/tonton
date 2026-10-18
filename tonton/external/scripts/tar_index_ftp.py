"""Walk a remote tar on FTP header-by-header using ranged reads; write index TSV (name, data_offset, size)."""
import ftplib, sys, socket
host, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
def connect():
    f = ftplib.FTP(host, timeout=60); f.login(); f.voidcmd('TYPE I'); return f
ftp = connect()
total = ftp.size(path)
def read(off, n):
    global ftp
    for attempt in range(5):
        try:
            conn = ftp.transfercmd('RETR ' + path, rest=off)
            buf = b''
            while len(buf) < n:
                c = conn.recv(n - len(buf))
                if not c: break
                buf += c
            conn.close()
            try: ftp.voidresp()
            except ftplib.all_errors: pass
            return buf
        except (ftplib.all_errors, socket.error) as e:
            try: ftp.close()
            except Exception: pass
            ftp = connect()
    raise RuntimeError(off)
off = 0
with open(out, 'w') as fo:
    while off < total:
        h = read(off, 512)
        if len(h) < 512 or h == b'\0' * 512: break
        name = h[:100].rstrip(b'\0').decode()
        prefix = h[345:500].rstrip(b'\0').decode()
        if prefix: name = prefix + '/' + name
        size = int(h[124:136].strip(b'\0 ') or b'0', 8)
        typ = h[156:157]
        fo.write(f'{name}\t{off+512}\t{size}\t{typ.decode()}\n'); fo.flush()
        off += 512 + ((size + 511) // 512) * 512
print('done', off, total)
