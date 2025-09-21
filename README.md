Math with Gestures
Short Description
This project is an interactive AI application that uses hand gestures to perform mathematical operations. It leverages the device's webcam to detect hand movements in real time and uses the MediaPipe framework to interpret gestures, allowing for a hands-free, intuitive user experience.

Tech Stack
Python: The core programming language for the application.

MediaPipe: A framework used for real-time hand gesture recognition and tracking.

OpenCV: A library for real-time video capture and processing from a webcam.

Installation
The project is containerized using Docker, which ensures it can run on any machine without compatibility issues.

To run this project, you need to have Docker installed on your system.

1. Pull the Docker Image
You can pull the pre-built image from Docker Hub using the following command:

docker pull adarsh2721/mathwithgestures:latest

2. Run the Docker Container
The container needs access to your webcam to function. Use the following command to run the container, granting it access to your camera:

docker run --net=host -it adarsh2721/mathwithgestures:latest


![Dashboard Screenshot 1](running.png)
