ARG ISAACSIM_BASE_IMAGE_ARG=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION_ARG=5.0.0
FROM ${ISAACSIM_BASE_IMAGE_ARG}:${ISAACSIM_VERSION_ARG} AS base

# Re-declare ARGs after FROM to make them available in this stage
ARG ISAACSIM_VERSION_ARG=5.0.0
ARG PIP_INDEX_URL=https://artnj.zte.com.cn/artifactory/api/pypi/public-pypi-virtual/simple
ARG HTTP_PROXY=http://proxyhk.zte.com.cn:80
ARG HTTPS_PROXY=http://proxyhk.zte.com.cn:80
ARG NO_PROXY=localhost,127.0.0.*,.zte.com.cn,.artifactory.zte.com.cn

# Environment variables
ENV ISAACSIM_VERSION=${ISAACSIM_VERSION_ARG} \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Set proxy for build time only (won't persist in final image unless explicitly set)
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

SHELL ["/bin/bash", "-c"]

LABEL maintainer="RealMirror Team" \
      description="Dockerfile for building and running the RealMirror inside Isaac Sim container image." \
      version="${ISAACSIM_VERSION_ARG}"

# Replace Ubuntu sources with USTC mirrors and install dependencies in one layer
RUN cat > /etc/apt/sources.list <<'EOF' && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        linux-headers-generic \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# 默认注释了源码仓库,如有需要可自行取消注释
deb https://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse

deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse

# 预发布软件源,不建议启用
# deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
EOF

# Install Desktop Environment, VNC, and XRDP
ARG DESKTOP_ENV=xfce4
ENV DISPLAY=:1 \
    VNC_PORT=5901 \
    XRDP_PORT=3389

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Desktop Environment
        ${DESKTOP_ENV} \
        ${DESKTOP_ENV}-goodies \
        dbus-x11 \
        # VNC Server
        tigervnc-standalone-server \
        tigervnc-common \
        tigervnc-tools \
        # XRDP
        xrdp \
        xorgxrdp \
        # X Server
        xvfb \
        x11-xserver-utils \
        # Utilities
        xterm \
        supervisor \
        net-tools \
        locales \
        fonts-wqy-zenhei \
        fonts-wqy-microhei && \
    # Configure locale
    locale-gen en_US.UTF-8 && \
    # Setup VNC directory (password will be set at runtime)
    mkdir -p /root/.vnc && \
    # Setup XRDP
    sed -i 's/port=3389/port=3389/g' /etc/xrdp/xrdp.ini && \
    sed -i 's/max_bpp=32/max_bpp=24/g' /etc/xrdp/xrdp.ini && \
    echo "startxfce4" > /root/.xsession && \
    chmod +x /root/.xsession && \
    # Configure supervisor
    mkdir -p /var/log/supervisor && \
    # Cleanup
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create VNC startup script
RUN cat > /root/.vnc/xstartup <<'VNCEOF' && \
    chmod +x /root/.vnc/xstartup
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XKL_XMODMAP_DISABLE=1
export XDG_CURRENT_DESKTOP="XFCE"
export XDG_SESSION_TYPE="x11"
exec startxfce4
VNCEOF

# Create supervisor configuration
RUN cat > /etc/supervisor/conf.d/desktop.conf <<'SUPEOF'
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor

[program:vncserver]
command=/usr/bin/vncserver :1 -geometry 1920x1080 -depth 24 -localhost no -fg
autorestart=true
stdout_logfile=/var/log/supervisor/vncserver.log
stderr_logfile=/var/log/supervisor/vncserver_err.log
priority=100

[program:xrdp-sesman]
command=/usr/sbin/xrdp-sesman --nodaemon
autorestart=true
stdout_logfile=/var/log/supervisor/xrdp-sesman.log
stderr_logfile=/var/log/supervisor/xrdp-sesman_err.log
priority=200

[program:xrdp]
command=/usr/sbin/xrdp --nodaemon
autorestart=true
stdout_logfile=/var/log/supervisor/xrdp.log
stderr_logfile=/var/log/supervisor/xrdp_err.log
priority=300
SUPEOF

# Install Node.js via nvm
ARG NODE_VERSION=22.19.0
ARG NVM_VERSION=v0.40.1
ENV NVM_DIR=/root/.nvm \
    NODE_PATH=/root/.nvm/versions/node/v${NODE_VERSION}/lib/node_modules \
    PATH=/root/.nvm/versions/node/v${NODE_VERSION}/bin:$PATH

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh | bash && \
    . ${NVM_DIR}/nvm.sh && \
    nvm install ${NODE_VERSION} && \
    nvm use ${NODE_VERSION} && \
    nvm alias default ${NODE_VERSION} && \
    npm config set registry https://registry.npmmirror.com && \
    node --version && \
    npm --version

# Install Python dependencies
COPY . /workspace/zbot


RUN /isaac-sim/python.sh -m pip install --upgrade pip -i ${PIP_INDEX_URL} && \
/isaac-sim/python.sh -m pip freeze > /tmp/constraints.txt && \
/isaac-sim/python.sh -m pip install -e /workspace/zbot/thirdparty/lerobot/ --no-cache-dir -c /tmp/constraints.txt -i ${PIP_INDEX_URL} && \
/isaac-sim/python.sh -m pip install -e /workspace/zbot --no-cache-dir -c /tmp/constraints.txt -i ${PIP_INDEX_URL} && \
rm -rf /tmp/constraints.txt /root/.cache/pip

# Create startup script for desktop services
# Copy start-desktop.sh script and set execute permission
COPY docker/teleop/start-desktop.sh /usr/local/bin/start-desktop.sh
RUN chmod +x /usr/local/bin/start-desktop.sh

WORKDIR /workspace/zbot

ENTRYPOINT []

# Use start-desktop.sh script to keep container running
CMD ["/usr/local/bin/start-desktop.sh"]


