import random
import simpy
from statistics import mean

# Globals:metrics you will print out
total_cars = 0   # car counter
wash_times = []  # Array for time it takes to wash a car
dry_times = []   # Array for how long it takes to dry a car
tot_times = []   # Array for total time spent at the car wash
wash_queue= []
dry_queue = []
NUM_REPL = 5
"""
Create a car wash object to declare all entities that will move through the carwash and the actions we are interested in tracking in the environment:
Cars, wash (count/number), wash (duration/rate), dry(count, number), dry(duration/rate

The object declaration allows to see the objects that make up the simulation
"""

class Carwash(object):
    
    def __init__(self, env, count_wash, wash_dur, count_dry, dry_dur):
        # Definition of agents that will move through car wash, and actions
        
        self.env = env

        # Saving arguments/metrics as member variables
        self.count_wash = count_wash
        self.count_dry = count_dry
        self.wash_dur = wash_dur
        self.dry_dur = dry_dur 

        # Create queues: multiple wash queue, multiple dry queues
        self.washers = [] # Empty array
        for i in range(self.count_wash):
            self.washers.append(simpy.Resource(env,1)) # One resource for each washer
        self.dryers = []    # Empty array; one resource for each dryer (n-queues)
        for i in range(self.count_dry):
            self.dryers.append(simpy.Resource(env,1)) # For each dryer, add a single resource

    # Now create some functions to handle our processes, namely the wash process and dry process
    """ Functions allow us to see the actions that happen in the simulation: cars moving through queues, washing and drying"""


    # Car wash process defined to run for 10 minutes
    def wash_car(self, car_name):
        # Constant
        yield self.env.timeout(10)

    # Hand dry process; needs to be passed in an entity to operate on
    def dry_car(self, car_name):
        # Uniform
        yield self.env.timeout(random.uniform(self.dry_dur[0], self.dry_dur[1]))

# Car function - this defines how a car moves through the system;

def Car(env, name, carwash):
    # Modify some globals
    global wash_time
    global dry_time  
    global tot_time
    global wash_queue
    global dry_queue

    # We need to calculate all metrics that we want to track
    ta = env.now    # Note the time the car arrives, time it enters wash queue
    tw_wash = 0     # Wait time for a wash bay
    tva_wash = 0    # Time spent getting car washed
    tw_dry = 0      # Wait time for the dry bay
    tva_dry = 0     # Time spent getting car dried
    q_wash=0
    q_dry=0

    # Car first car randomly gets assigned to one of the wash bays
    # Just need to 'request' (SimPy) a resource
    wash_bay = random.randint(0,carwash.count_wash-1)
    with carwash.washers[wash_bay].request() as request:

        q_wash =len(wash_bay.queue)
        # Request a wash bay
        yield request

        # Note the start time
        ts = env.now

        # Wait time is start time - arrival time
        tw_wash = ts - ta
        
        # Wash the car
        yield env.process(carwash.wash_car(name))

        # Save the time spent in the wash queue
        tva_wash = env.now-ts

    

    ta_dry = env.now 

    # Now car randomly gets assigned to one of the dry bays
    dry_bay = random.randint(0,carwash.count_dry-1)
    with carwash.dryers[dry_bay].request() as request:
        q_dry=len(dry_bay.queue)
        yield request
        ts = env.now
        tw_dry = ts - ta_dry
        yield env.process(carwash.dry_car(name))
        tva_dry = env.now-ts
    
    # Cars can technically leave now, but we want to save some metrics before they do
    wash_times.append([tw_wash, tva_wash])
    dry_times.append([tw_dry, tva_dry])
    tot_times.append(env.now - ta)
    wash_queue.append(q_wash)
    dry_queue.append(q_dry)



# This function defines how cars arrive 
def Cars_Arrive(env, carwash, arr_rate):
    # globals
    global total_cars

    # Internal tracker for each replication
    count_cars = 0

    while True:
        # Simulate arrival time
        yield env.timeout(random.expovariate(arr_rate))

        # Increment the number of cars
        count_cars += 1
        total_cars += 1

        # After simulating the arrival time, create a car using Simpy "process"
        env.process( Car(env, f'Car {count_cars}', carwash))


# Run a single simulation; just wrap these lines in a loop to do multiple replciations
#####
#####

# Create a SimPy environment
env = simpy.Environment()

# Create an carwash based off  def __init__(self, env, count_wash, wash_dur, count_dry, dry_dur):
# 4 wash bays that take a constant 8 minutes per wash
# 6 dry bays, whose dry duration ranges between 3 and 7 minutes
carwash = Carwash(env, 4, 8, 6, [3,7])

# Start the simulation by having cars arrive at rate of 1 car every 2 minutes
# Car arrival rate is 0.5
env.process(Cars_Arrive(env, carwash, .5))

# Run the simulation for 3 hours
env.run(3*60)

#####
#####

# Print some stats
print(f'Average number of cars {total_cars/NUM_REPL}')
avg_wash_queue = mean(list(zip(*wash_queue))[0])
avg_wash_duration = mean(list(zip(*wash_times))[1])
avg_dry_queue = mean(list(zip(*dry_queue))[0])
avg_dry_duration = mean(list(zip(*dry_times))[1])
avg_sys_time = mean(tot_times)
avg_queue_time = avg_wash_wait + avg_dry_wait
print(f'Average wash queue length {avg_wash_queue}')
print(f'Average washing duration {avg_wash_duration}')
print(f'Average dry queue length {avg_dry_queue}')
print(f'Average drying duration {avg_dry_duration}')
print(f'Average time spent at carwash {avg_sys_time}')
print(f'Average wait time {avg_wait_time}')
print(f'Avereage VA time {avg_wash_time + avg_dry_time}')
