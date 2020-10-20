class Track:

    """
    Track object holds the racetrack as a binary array,
    and the possible spawning positions as a n x 2 numpy array.
    """

    def __init__(self, base_image, spawn_at):

        """
        Convert RGB to Grey-Scale.
        Image expected to be of n x m
        """
        
        res=base_image.astype('int16')
        self.track=res
        self.spawn_at=spawn_at

        return 