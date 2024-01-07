from algocert.init_set.init_set import InitSet


class ConstShiftParamSet(InitSet):

    def __init__(self, y, x, shift):
        '''
            set for y = x + c, where y, x both params
        '''
        super().__init__(y)
        self.y = y
        self.x = x
        self.shift = shift

    def __str__(self):
        to_string = f'SET({self.y.name}): ConstShiftParamSet = {self.x.name}, {self.shift.reshape(-1,)}'
        return to_string
