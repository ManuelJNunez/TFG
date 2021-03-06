#!/bin/bash
yum update -y
yum install -y java-11-amazon-corretto
wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo
rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io.key
yum update -y
yum install -y jenkins git docker
usermod -aG docker ec2-user
systemctl enable jenkins
systemctl start jenkins
systemctl enable docker
systemctl start docker
yum install nftables -y
nft add table nat
nft add chain ip nat prerouting { type nat hook prerouting  priority 0 \; }
nft add chain ip nat postrouting { type nat hook postrouting priority 100 \; }
nft add rule nat prerouting tcp dport 80 redirect to 8080
