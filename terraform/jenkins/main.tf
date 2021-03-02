provider "aws" {
  region = "us-east-1"
}

variable "your_ip" {
  type        = string
  description = "Ingresa tu IP en formato CIDR para poder configurar el ingreso por SSH."
}

resource "aws_security_group" "jenkins_sg" {
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
    Name = "Jenkis SG"
  }
}

resource "aws_instance" "jenkins" {
  instance_type   = "t2.micro"
  ami             = "ami-047a51fa27710816e"
  key_name        = "m1"
  security_groups = [aws_security_group.jenkins_sg.name]

  user_data = file("provision.sh")

  tags = {
    Name = "Jenkins Instance"
  }
}

resource "aws_eip" "lb" {
  instance = aws_instance.jenkins.id
}
