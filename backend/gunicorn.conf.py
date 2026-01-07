import multiprocessing

# Bind to localhost, nginx will proxy
bind = "127.0.0.1:8000"

# Workers
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/ai_data_analyst/gunicorn_access.log"
errorlog = "/var/log/ai_data_analyst/gunicorn_error.log"
loglevel = "info"

# Process naming
proc_name = "ai_data_analyst"

# Server mechanics
daemon = False
pidfile = "/run/ai_data_analyst/gunicorn.pid"
user = "www-data"
group = "www-data"
tmp_upload_dir = None

# SSL (handled by nginx)
keyfile = None
certfile = None
