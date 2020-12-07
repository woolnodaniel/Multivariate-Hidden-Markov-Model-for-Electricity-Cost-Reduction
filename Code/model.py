"""
Creates all variables to define the probabilistic graphical model as described in the report
"""

# Adj: adjacency matrix: edges from each room to its neighbours (including itself)
Adj = {
    'r1':['r1','r2','r3'],
    'r2':['r1','r2','r4'],
    'r3':['r1','r3','r7'],
    'r4':['r2','r4','r8'],
    'r5':['r5','r6','r9','c3a'],
    'r6':['r5','r6','c3a'],
    'r7':['r3','r7','c1'],
    'r8':['r4','r8','r9'],
    'r9':['r5','r8','r9','r13'],
    'r10':['r10','c3a'],
    'r11':['r11','c3a'],
    'r12':['r12','r22','outside'],
    'r13':['r9','r13','r24'],
    'r14':['r14','r24'],
    'r15':['r15','c3a'],
    'r16':['r16','c3a'],
    'r17':['r17','c3b'],
    'r18':['r18','c3b'],
    'r19':['r19','c3b'],
    'r20':['r20','c3b'],
    'r21':['r21','c3b'],
    'r22':['r12','r22','r25'],
    'r23':['r23','r24'],
    'r24':['r13','r14','r23','r24'],
    'r25':['r22','r25','r26','c1'],
    'r26':['r25','r26','r27'],
    'r27':['r26','r27','r32'],
    'r28':['r28','c4'],
    'r29':['r29','r30','c4'],
    'r30':['r29','r30'],
    'r31':['r31','r32'],
    'r32':['r27','r31','r32','r33'],
    'r33':['r32','r33'],
    'r34':['r34','c2'],
    'r35':['r35','c4'],
    'c1':['r7','r25','c1','c2'],
    'c2':['r34','c1','c2','c4'],
    #'c3':['r5','r6','r10','r11','r15','r16','r17','r18','r19','r20','r21','c3','o1'],
    'c3a': ['r5', 'r6', 'r10', 'r11', 'r15', 'r16', 'c3a', 'c3b'],
    'c3b': ['r17', 'r18', 'r19', 'r20', 'r21', 'c3a', 'c3b', 'o1'],
    'c4':['r28','r29','r35','c2','c4','o1'],
    'o1':['c3b','c4','o1'],
    'outside':['r12','outside']
}

# Areas: all rooms
Areas = list(Adj.keys())

# Rooms: only those rooms beginning with 'r'
Rooms = [R for R in Areas if R[0] == 'r']

# Sig: adjacency matrix between sensors and the rooms they can detect motion in
Sig = {
    'reliable_sensor1': ['r16'],
    'reliable_sensor2': ['r5'],
    'reliable_sensor3': ['r25'],
    'reliable_sensor4': ['r31'],
    'unreliable_sensor1': ['o1'],
    'unreliable_sensor2': ['c3a', 'c3b'],
    'unreliable_sensor3': ['r1'], #r4???
    'unreliable_sensor4': ['r24'],
    'door_sensor1': ['r8', 'r9'],
    'door_sensor2': ['c1', 'c2'],
    'door_sensor3': ['r26', 'r27'],
    'door_sensor4': ['r35', 'c4']
}

# Sensors: stationary sensors
Sensors = Sig.keys()

# Wandering Robots:
Wandering_Robots = ['robot1', 'robot2']

# outcomeSpace: after mapping our dataframe (see learn_parameter.py),
# all room and sensor variables have outcome space 0 and 1.
# For room: 1 indicates occupied, 0 indicates empty
# For sensors: 1 indicates motion detected, 0 indicates no motion detected.
outcomeSpace = {R:(0,1) for R in set(Sensors) | set(Wandering_Robots)}
for R in Areas:
    outcomeSpace[R] = (0,1)
    outcomeSpace[R + '^t'] = (0,1)
    outcomeSpace[R + '^t+1'] = (0,1)