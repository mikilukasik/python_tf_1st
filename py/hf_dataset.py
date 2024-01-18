import time
from datasets import load_dataset

lines_to_get = 25000

# Start timer
start_time = time.time()

# Load dataset
dataset = load_dataset("laion/strategic_game_chess", 'en', streaming=True)

# Print time taken
print("Time to load dataset:", time.time() - start_time)

# Start new timer
start_time = time.time()

# Access train dataset
train_dataset = dataset["train"]

# Print time taken
print("Time to access train dataset:", time.time() - start_time)

# Start new timer
start_time = time.time()

# Create an iterable
iterable = iter(train_dataset)

# Print time taken
print("Time to create iterable:", time.time() - start_time)

loop_start_time = time.time()

# Repeat for each print statement
for _ in range(lines_to_get):
    start_time = time.time()
    a = next(iterable)
    # print(next(iterable))
    # print("Time for this operation:", time.time() - start_time)

print("Total time for ", lines_to_get, " lines:", time.time() - loop_start_time)
