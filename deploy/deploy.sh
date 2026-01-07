#!/bin/bash
set -e

# Configuration
APP_NAME="ai_data_analyst"
APP_DIR="/var/www/$APP_NAME"
REPO_URL="your-repo-url"  # Change this
DOMAIN="your-domain.com"  # Change this

echo "=== AI Data Analyst Production Deployment ==="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./deploy.sh)"
    exit 1
fi

# Update system
echo "[1/10] Updating system packages..."
apt update && apt upgrade -y

# Install dependencies
echo "[2/10] Installing dependencies..."
apt install -y python3 python3-pip python3-venv nodejs npm nginx certbot python3-certbot-nginx git

# Create app directory
echo "[3/10] Setting up application directory..."
mkdir -p $APP_DIR
mkdir -p /var/log/$APP_NAME
chown -R www-data:www-data /var/log/$APP_NAME

# Clone or pull repository
if [ -d "$APP_DIR/.git" ]; then
    echo "[4/10] Pulling latest changes..."
    cd $APP_DIR
    git pull
else
    echo "[4/10] Cloning repository..."
    git clone $REPO_URL $APP_DIR
    cd $APP_DIR
fi

# Backend setup
echo "[5/10] Setting up backend..."
cd $APP_DIR/backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "[!] Creating .env file - EDIT THIS WITH YOUR VALUES"
    cat > .env << EOF
DEBUG=False
DJANGO_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(50))')
ALLOWED_HOSTS=$DOMAIN,www.$DOMAIN
CORS_ALLOWED_ORIGINS=https://$DOMAIN,https://www.$DOMAIN
OPENAI_API_KEY=your-openai-api-key
# DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
EOF
    echo "[!] IMPORTANT: Edit /var/www/$APP_NAME/backend/.env with your values!"
fi

# Run migrations and collect static
python manage.py migrate --noinput
python manage.py collectstatic --noinput

deactivate

# Frontend setup
echo "[6/10] Building frontend..."
cd $APP_DIR/frontend
npm install
npm run build

# Set permissions
echo "[7/10] Setting permissions..."
chown -R www-data:www-data $APP_DIR

# Setup systemd
echo "[8/10] Configuring systemd..."
cp $APP_DIR/deploy/ai_data_analyst.service /etc/systemd/system/
cp $APP_DIR/deploy/ai_data_analyst.socket /etc/systemd/system/
mkdir -p /run/$APP_NAME
chown www-data:www-data /run/$APP_NAME

systemctl daemon-reload
systemctl enable $APP_NAME.socket
systemctl enable $APP_NAME.service
systemctl restart $APP_NAME.socket
systemctl restart $APP_NAME.service

# Setup nginx
echo "[9/10] Configuring nginx..."
# Update nginx config with actual domain
sed "s/your-domain.com/$DOMAIN/g" $APP_DIR/deploy/nginx.conf > /etc/nginx/sites-available/$APP_NAME
ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl restart nginx

# SSL setup
echo "[10/10] Setting up SSL..."
echo "Run the following command to get SSL certificate:"
echo "  sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit /var/www/$APP_NAME/backend/.env with your OPENAI_API_KEY"
echo "2. Run: sudo certbot --nginx -d $DOMAIN"
echo "3. Uncomment HTTPS config in nginx and restart nginx"
echo ""
echo "Useful commands:"
echo "  - Check status: systemctl status $APP_NAME"
echo "  - View logs: journalctl -u $APP_NAME -f"
echo "  - Restart: systemctl restart $APP_NAME"
