#!/bin/bash

# Java and Jenkins Install
apt update
apt install openjdk-11-jdk -y
wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | apt-key add -
sh -c 'echo deb https://pkg.jenkins.io/debian-stable binary/ > \
    /etc/apt/sources.list.d/jenkins.list'
apt-get update
apt-get install jenkins -y


# Docker Install
apt-get remove docker docker-engine docker.io containerd runc -y
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install docker-ce docker-ce-cli containerd.io -y
usermod -aG docker ubuntu
usermod -aG docker jenkins

# Enable and start services
systemctl enable jenkins
systemctl start jenkins
systemctl enable docker
systemctl start docker

# Redirect port 80 to 8080
apt-get install nftables -y
nft add table nat
nft add chain ip nat prerouting { type nat hook prerouting  priority 0 \; }
nft add chain ip nat postrouting { type nat hook postrouting priority 100 \; }
nft add rule nat prerouting tcp dport 80 redirect to 8080
