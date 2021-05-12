provider "aws" {
  region = "us-east-1"
}

variable "your_ip" {
  type        = string
  description = "Ingresa tu IP en formato CIDR para poder configurar el ingreso por SSH."
}

resource "aws_security_group" "mlflow_sg" {
  ingress {
    description      = "Allow HTTP traffic"
    from_port        = 80
    to_port          = 80
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  ingress {
    description = "Allow SSH traffic"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.your_ip]
  }

  egress {
    cidr_blocks = [ "0.0.0.0/0" ]
    ipv6_cidr_blocks = ["::/0"]
    from_port = 0
    protocol = "-1"
    to_port = 0
  }

  tags = {
    Name = "MLFlow SG"
  }
}


resource "aws_instance" "mlflow" {
  instance_type   = "t2.small"
  ami             = "ami-042e8287309f5df03"
  key_name        = "m2"
  security_groups = [aws_security_group.mlflow_sg.name]

  provisioner "file" {
    source      = "Dockerfile"
    destination = "/home/ubuntu/Dockerfile"
    
    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("../keys/m2.pem")
      host        = self.public_dns
    }
  }

  provisioner "file" {
    source      = "docker-compose.yml"
    destination = "/home/ubuntu/docker-compose.yml"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("../keys/m2.pem")
      host        = self.public_dns
    }
  }

  provisioner "file" {
    source      = ".env"
    destination = "/home/ubuntu/.env"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file("../keys/m2.pem")
      host        = self.public_dns
    }
  }

  user_data = file("provision.sh")

  tags = {
    Name = "MLFlow Instance"
  }
}


resource "aws_eip" "lb" {
  instance = aws_instance.mlflow.id
}