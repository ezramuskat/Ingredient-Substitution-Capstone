#!/bin/bash

# Start backend
flask --app app.py --debug run &
FLASK_PID=$!

# Start frontend
npm run --prefix frontend dev &
REACT_PID=$!

# Trap to kill both processes on exit
trap "kill $FLASK_PID $REACT_PID" SIGINT SIGTERM EXIT

# Wait for either to exit
wait $FLASK_PID
wait $REACT_PID