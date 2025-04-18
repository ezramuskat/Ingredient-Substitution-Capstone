#!/bin/bash

cd frontend
npm install
# Start frontend
npm run build &
REACT_PID=$!
cd ..

# Start backend
flask --app app.py run -h 0.0.0.0 -p 5000 &
FLASK_PID=$!

# Trap to kill both processes on exit
trap "kill $FLASK_PID $REACT_PID" SIGINT SIGTERM EXIT

# Wait for either to exit
wait $FLASK_PID
wait $REACT_PID