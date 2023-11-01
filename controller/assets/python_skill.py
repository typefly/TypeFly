def scan(object_name):
    for i in range(4):
        var_1 = obj_loc_y(object_name)
        if var_1 > 0.6:
            move_down(20)
        if var_1 < 0.4:
            move_up(20)
        var_2 = obj_loc_y(object_name)
        if var_2 < 0.6 and var_2 > 0.4:
            return True
    return False

def find_an_apple():
    orienting('apple')
    approach('apple')

def turn_to_animal():
    for i in range(8):
        if query('is there any animal') == True:
            return True
        turn_left(45)
    return False
