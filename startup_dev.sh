#!/bin/bash

# Start backend
export FLASK_ENV=production
flask --app app.py --debug run &
FLASK_PID=$!

npm install --prefix frontend
# Start frontend
npm run --prefix frontend dev &
REACT_PID=$!

# Trap to kill both processes on exit
trap "kill $FLASK_PID $REACT_PID" SIGINT SIGTERM EXIT

# Wait for either to exit
wait $FLASK_PID
wait $REACT_PID