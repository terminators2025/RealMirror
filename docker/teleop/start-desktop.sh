#!/bin/bash
set -e

# Set VNC password from environment variable or use default
VNC_PASSWORD=${VNC_PASSWORD:-realmirror}
mkdir -p /root/.vnc

# Create VNC password file using vncserver's built-in password prompt
# We pipe the password twice (password + confirmation)
echo "Setting up VNC password..."
printf '%s\n%s\n' "$VNC_PASSWORD" "$VNC_PASSWORD" | vncserver -localhost no :1 2>&1 || true

# Kill the temporary VNC server that was just started
vncserver -kill :1 2>/dev/null || true

# Verify password file exists
if [ ! -f /root/.vnc/passwd ]; then
    echo "ERROR: Failed to create VNC password file"
    exit 1
fi

chmod 600 /root/.vnc/passwd
echo "VNC password configured successfully"

# Clean any existing VNC locks
rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true

# Start VNC server directly (without supervisor for now)
echo "Starting VNC server on display :1..."
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

# Start XRDP services
echo "Starting XRDP services..."
/usr/sbin/xrdp-sesman &
/usr/sbin/xrdp

echo "Desktop services started successfully!"
echo "VNC available on port 5901"
echo "XRDP available on port 3389"

# Keep the script running
tail -f /var/log/xrdp.log /var/log/xrdp-sesman.log /root/.vnc/*.log 2>/dev/null

# Add container running message loop
while true; do 
    echo 'Container is running...'
    sleep 60
done