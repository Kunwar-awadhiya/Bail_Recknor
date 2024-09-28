## Bail Rockner

Bail Reckoner is a digital solution designed to streamline the bail process for undertrial prisoners, legal aid providers, and judicial authorities. The project was developed as part of Smart India Hackathon (SIH) 2024 with the goal of providing an easy-to-use platform that integrates various legal parameters to assess bail eligibility.

This application is built with the MERN stack (MongoDB, Express.js, React.js, Node.js) and leverages modern technologies for real-time updates, form handling, and data visualization.

## Features

Bail Eligibility Assessment: Streamlines the evaluation of bail eligibility based on legal parameters like the nature of the offense, duration of imprisonment, and judicial discretion.
Real-Time Updates: Offers real-time updates for bail application progress using Socket.io.
Tracking Dashboards: Provides tracking dashboards for legal professionals and judicial authorities to manage bail applications efficiently.
User-Friendly Interface: A responsive, interactive frontend built with React.js and styled using Tailwind CSS (or Bootstrap).
Backend Logic: Backend API built using Node.js and Express.js, supporting the bail assessment workflow and integration with the AMR classifier.

## Tech Stack

Frontend:
React.js
Tailwind CSS or Bootstrap
Axios for API communication
Redux or Context API for state management
Formik or React Hook Form for form handling
Socket.io for real-time updates.

Backend:
Node.js
Express.js
MongoDB for database
Python AI/ML classifier integration

## Installation

Clone the repository:
Copy code
git clone https://github.com/yourusername/bail-reckoner.git
cd bail-reckoner
Install dependencies for both frontend and backend:

# Frontend
cd client
npm install

# Backend
cd ../server
npm install
Set up environment variables: Create a .env file in the server directory and add the following:

MONGO_URI=<your-mongodb-connection-string>
PORT=5000
Run the application:

# Start backend server
cd server
npm start

# Start frontend server
cd ../client
npm start
The frontend will run on http://localhost:3000 and the backend on http://localhost:5000.
    
## Usage/Examples

Open the application in your browser at http://localhost:3000.
Use the interface to input necessary legal parameters for bail assessment.
View real-time updates and tracking of bail applications.
Manage bail-related data with admin and legal aid provider access.

## Contributing

We welcome contributions to improve Bail Reckoner! Please follow the steps below:

Fork the repository.
Create a new branch for your feature (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add a new feature').
Push to the branch (git push origin feature-branch).
Create a pull request.

