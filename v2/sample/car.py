import numpy as np

class Car:
    """
    The car object has the following attributes:
    1. Track - the associated track
    2. current position: Current position on the track. Floating point coordinate.
    3. integer_position_: defined as current_position.astype('int16')
    4. Bool collided_on_last
    5. sensors: LIDAR type function
    6. init_speed_
    7. max_speed
    8. sampling frequency: The frequency of play.
    9. speed: Current speed
    10. car_angle: current car angle as define below.
    """ 
    
    
   
    def __init__(self, track, max_speed, sampling_frequency):
        
        """
        track: The track as a track object
        max_speed: Max attainable speed
        sampling_frequency:
        
        car angle is defined as the angle of the (unimplemented) velocity vector from
        the standard mathematical X axis, i.e., 
        y ^
          |
          |
          |
          -------> x
        Positive is ACW.
        """

        self.track=track
        self.last_checkpoint=self.track.spawn_at[np.random.choice(range(len(self.track.spawn_at)))]
        self.next_checkpoint=self.track.next_checkpoint(self.last_checkpoint)
        self.time_in_current_sector=0
        
        # self.possible_position=spawn_at
        self.current_position=self.last_checkpoint
        self.integer_position_=self.current_position.astype('int16')
        self.collided_on_last=None
    
        self.init_speed__=0
        self.speed=self.init_speed__
        self.max_speed=max_speed
        self.sampling_frequency=sampling_frequency

        self.car_angle=np.random.uniform(0, 1)*2*np.pi
        
        self.sensors=np.zeros((8,))
        self.load_sensors()


        return
    
    def duplicate(self):

        """
        Returns: A car object, completely mimicing the current car object in every way.
        """

        car=Car(self.track, self.max_speed)
        car.last_checkpoint=np.zeros((2, ))+self.last_checkpoint
        car.next_checkpoint=car.track.next_checkpoint(car.last_checkpoint)
        car.time_in_current_sector=self.time_in_current_sector

        car.current_position=car.current_position*0+self.current_position
        car.integer_position_=car.current_position.astype('int16')
        car.collided_on_last=self.collided_on_last
        
        car.speed=self.speed
        car.car_angle=self.car_angle
        
        car.sampling_frequency=self.sampling_frequency
        car.load_sensors()
        
        return car
        
    
    def re_spawn(self):

        """
        Re-spawn the current car at a legal position. Consult spawn_at attribute of associated track.
        Set all car state attributes.
        """
        
        self.speed=self.init_speed__
        self.last_checkpoint=self.track.spawn_at[np.random.choice(range(len(self.track.spawn_at)))]
        self.next_checkpoint=self.track.next_checkpoint(self.last_checkpoint)
        self.time_in_current_sector=0

        self.current_position=self.last_checkpoint
        self.integer_position_=self.current_position.astype('int16')
        self.collided_on_last=None
        self.load_sensors()
        self.speed=self.init_speed__
        self.car_angle=np.random.uniform(0, 1)*2*np.pi
        return        
                
    def execute_forces(self, f1, f2, max_magnitudes=50):

        """
        Execute the forces.
        Update car state attributes:
            speed
            car_angle
            collided_on_last
            current_position, integer_position
            sensors

        f1 is the force in the vertical direction
        f2 is the force in the horizrontal direction

        ^
        |
        |     this is f1


        ------>  this is f2
        
        f1 is expected between -1, 1
        f2 is expected between -1, 1
        """

        f1=max_magnitudes*f1
        f2=max_magnitudes*f2


        if self.speed==0:
            if f1==0 or f2==0:
                if f1==0 and f2==0:
                    self.time_in_current_sector=self.time_in_current_sector+1.0/self.sampling_frequency
                    return
                elif f1!=0 and f2==0:
                    self.car_angle=np.pi/2
                    if f1<0:
                        self.car_angle=3*np.pi/2
                else:
                    self.car_angle=0
                    if f2<0:
                        self.car_angle=self.car_angle+np.pi
            else:
                abs_angle=np.arctan(abs(f1/f2))
                if f1>0 and f2>0:
                    self.car_angle=abs_angle
                elif f1>0 and f2<0:
                    self.car_angle=np.pi-abs_angle
                elif f1<0 and f2<0:
                    self.car_angle=np.pi+abs_angle
                else:
                    self.car_angle=2*np.pi-abs_angle
                
        
        self.speed=min(
            abs(self.speed+(f2*np.cos(self.car_angle)+f1*np.sin(self.car_angle))*(1.0/self.sampling_frequency)),
            self.max_speed
        )
        
        delta_angle=0
        if self.speed!=0:
            delta_angle=np.arctan( (f1*np.cos(self.car_angle)-f2*np.sin(self.car_angle))
                /
                (self.speed*self.sampling_frequency)
                )
        self.car_angle=np.mod(self.car_angle+delta_angle, 2*np.pi)

        movement=np.asarray([
            -1*self.speed*np.sin(self.car_angle)*1.0/self.sampling_frequency, 
            self.speed*np.cos(self.car_angle)*1.0/self.sampling_frequency])
        if max(abs(movement))==0:
            print("Zero Movement recorded. Speed is: ", self.speed, " sampling_frequency is: ", self.sampling_frequency)

        old_position=np.zeros((2,))+self.current_position
        old_int_position=np.zeros((2,))+self.integer_position_
        self.current_position=self.current_position+movement
        self.integer_position_=self.current_position.astype('int16')
        cond=(
            np.min(self.current_position)<0 or
            (self.current_position>=self.track.track.shape).any() or
            self.track.track[self.integer_position_[0], self.integer_position_[1]]!=1
        )
        if cond:
            for distance in range(0, int(np.ceil(self.max_speed*self.sampling_frequency))+1):
                movement=np.asarray([-distance*np.sin(self.car_angle), distance*np.cos(self.car_angle)])
                temp_pos=old_int_position+movement
                temp_pos_int=temp_pos.astype('int16')
                if min(temp_pos_int)>=0 and (temp_pos_int<self.track.track.shape).all():
                    if self.track.track[temp_pos_int[0], temp_pos_int[1]]==1:
                        self.current_position=self.current_position*0+temp_pos
                        self.integer_position_=self.current_position.astype('int16')
                    else:
                        break
                else:
                    break
            self.speed=0
            self.collided_on_last=True
        else:
            self.collided_on_last=False
        self.load_sensors()

        ## ------------- timing calculation---------
        dx, dy=self.current_position[0]-self.next_checkpoint[0], self.current_position[1]-self.next_checkpoint[1]
        cnd=(
            np.sqrt(dx**2+dy**2)<self.track.min_checkpoint_distance and
            (old_position[1]<self.next_checkpoint[1] and self.current_position[1]>=self.next_checkpoint[1]
                or
            old_position[1]>self.next_checkpoint[1] and self.current_position[1]<=self.next_checkpoint[1]
            )
            )
        if cnd:
            self.last_checkpoint=self.next_checkpoint
            self.next_checkpoint=self.track.next_checkpoint(self.last_checkpoint)
            self.time_in_current_sector=0
        else:
            self.time_in_current_sector=self.time_in_current_sector+1.0/self.sampling_frequency

        return


    def load_sensors(self):

        """
        sensors will be at 
        0, 30, 60, 90, -30
        -60, -90, 180 (directly backward)
        degrees
        """
        
        angles=[0, np.pi/6, np.pi/3, np.pi/2, -np.pi/6, -np.pi/3, -np.pi/2, np.pi]
        temp_data=np.zeros((8, ))
        for angle_index in range(len(angles)):
            cur_angle=np.mod(self.car_angle+angles[angle_index], 2*np.pi)
            for distance in range(1, 101):
                r, c=int(self.integer_position_[0]-distance*np.sin(cur_angle)), int(self.integer_position_[1]+distance*np.cos(cur_angle))
                if min(r, c)<0 or r>=self.track.track.shape[0] or c>=self.track.track.shape[1]:
                    temp_data[angle_index]=distance-1
                    break
                if self.track.track[r, c]==0:
                    temp_data[angle_index]=distance-1
                    break
                if distance==100:
                    temp_data[angle_index]=distance
        self.sensors[0]=temp_data[7]
        self.sensors[4]=temp_data[0]
        self.sensors[5]=temp_data[1]
        self.sensors[6]=temp_data[2]
        self.sensors[7]=temp_data[3]
        self.sensors[1]=temp_data[6]
        self.sensors[2]=temp_data[5]
        self.sensors[3]=temp_data[4]
        return
