import numpy as np

class Track:

    """
    Track object has the following attributes:
    track: The racetrack as a 2D binary numpy array
    min_checkpoint_distance: The minimum distance that two checkpoints must adhere to
    spawn_at: The legal positions where a car can spawn at. Is the same as the checkpoints list, here.
    checkpoints: Mini-Sector breaking points
    """

    def __init__(self, base_image, min_checkpoint_distance):

        """
        Initializes the object.
        Parameters:
            base_image: The track as a 2D binary numpy array
            min_checkpoint_distance: The minimum distance that any 2 checkpoints must adhere to.
        """
        
        res=base_image.astype('int16')
        self.track=res
        self.min_checkpoint_distance=min_checkpoint_distance
        self.spawn_at=[]
        self.checkpoints=[]
        return

    def add_checkpoints(self, i,j):
        """
        Adds i,j to the checkpoints attribute, assuming that user takes care
        to input in correct order based on occurence of these points on the track.
        Checkpoints will allow to check if the car crosses the vertical line crossing the track at i,j.
        If the car has current_j<j and next_j>j (where j is the column index), then we know that it just went past.
        Obviously this restricts the choice of checkpoints to track positions where track goes left-right
        and not top-bottom.
        
        spawn_at will be equal to checkpoints always.

        Parameters:
            i: The row
            j: The column
        Returns:
            None. Updates self.checkpoints attribute.
        """

        minD=float('inf')
        for x,y in self.checkpoints:
            minD=min(np.sqrt((i-x)**2+(j-y)**2), minD)
        if minD<self.min_checkpoint_distance:
            raise RuntimeError("Checkpoint must be separated by at least {} pixels away".format(str(self.min_checkpoint_distance), ))

        self.checkpoints.append((i,j))
        self.spawn_at=self.checkpoints

    def next_checkpoint(self, last_Cp):
        """
        last_Cp=[i,j] is expected to be a checkpoint
        """
        i, j = last_Cp
        for idx, (i1, j1) in enumerate(self.checkpoints):
            if i1==i and j1==j:
                return self.checkpoints[int((idx+1)%len(self.checkpoints))]
