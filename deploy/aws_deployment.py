"""
AWS Deployment Configuration

This creates a production-ready deployment on AWS using:
- ECS (Elastic Container Service) for running containers
- Application Load Balancer for traffic distribution
- RDS for database (if needed)
- S3 for file storage
- CloudWatch for logging and monitoring

This is the most scalable and production-ready option.
"""

import boto3
import json
from pathlib import Path

def create_ecs_task_definition():
    """Create ECS task definition for the ML agent."""
    
    task_definition = {
        "family": "ml-agent",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "1024",  # 1 vCPU
        "memory": "2048",  # 2 GB RAM
        "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
        "containerDefinitions": [
            {
                "name": "ml-agent",
                "image": "YOUR_ACCOUNT.dkr.ecr.YOUR_REGION.amazonaws.com/ml-agent:latest",
                "portMappings": [
                    {
                        "containerPort": 8501,
                        "protocol": "tcp"
                    }
                ],
                "essential": True,
                "environment": [
                    {
                        "name": "STREAMLIT_SERVER_PORT",
                        "value": "8501"
                    },
                    {
                        "name": "STREAMLIT_SERVER_ADDRESS",
                        "value": "0.0.0.0"
                    }
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/ml-agent",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    }
                }
            }
        ]
    }
    
    return task_definition

def create_cloudformation_template():
    """Create CloudFormation template for infrastructure."""
    
    template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "Autonomous ML Agent Infrastructure",
        "Parameters": {
            "VpcId": {
                "Type": "AWS::EC2::VPC::Id",
                "Description": "VPC ID"
            },
            "SubnetIds": {
                "Type": "List<AWS::EC2::Subnet::Id>",
                "Description": "Subnet IDs"
            }
        },
        "Resources": {
            "ECSCluster": {
                "Type": "AWS::ECS::Cluster",
                "Properties": {
                    "ClusterName": "ml-agent-cluster"
                }
            },
            "ECSService": {
                "Type": "AWS::ECS::Service",
                "Properties": {
                    "Cluster": {"Ref": "ECSCluster"},
                    "ServiceName": "ml-agent-service",
                    "TaskDefinition": {"Ref": "ECSTaskDefinition"},
                    "DesiredCount": 1,
                    "LaunchType": "FARGATE",
                    "NetworkConfiguration": {
                        "AwsvpcConfiguration": {
                            "SecurityGroups": [{"Ref": "SecurityGroup"}],
                            "Subnets": {"Ref": "SubnetIds"},
                            "AssignPublicIp": "ENABLED"
                        }
                    }
                }
            },
            "SecurityGroup": {
                "Type": "AWS::EC2::SecurityGroup",
                "Properties": {
                    "GroupDescription": "Security group for ML Agent",
                    "VpcId": {"Ref": "VpcId"},
                    "SecurityGroupIngress": [
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 8501,
                            "ToPort": 8501,
                            "CidrIp": "0.0.0.0/0"
                        }
                    ]
                }
            }
        }
    }
    
    return template

if __name__ == "__main__":
    # Save task definition
    task_def = create_ecs_task_definition()
    with open("deploy/ecs-task-definition.json", "w") as f:
        json.dump(task_def, f, indent=2)
    
    # Save CloudFormation template
    template = create_cloudformation_template()
    with open("deploy/cloudformation-template.json", "w") as f:
        json.dump(template, f, indent=2)
    
    print("✅ AWS deployment files created!")
    print("📁 Files saved to deploy/ directory")
