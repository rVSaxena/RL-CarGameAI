This is a Deep Neural Network based Reinforcement Learning program that successfully
plays game_v1.

the game: 
A numpy based simple physics game where a point unit mass is to be navigated on a course.
The game ball starts at a random legal position, with zero speed, and after asking the user to specify the sampling frequency.
After this, for each play of the game, the user supplies the forces to be applied, in the cartesian coordinate system, 
and the game updates the position and state of the ball.
The 'Car' object plays the role of the said ball, likewise the 'Track' of the course.
Car object has a boolean attribute collided_on_last, which return True if the 'car' collided on the last move

For v1, the environment was retricted so as to
1. Have the ball maintain a constant speed.
2. Apply forces in directions either parallel or transverse to the direction of the velocity of the car.

The aim for the agent was to navigate at this constant speed without colliding.

Training:
500 epochs on track_pic5
99 epochs on track_pic1

Tested:
On track_pic9