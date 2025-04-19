# Self Driving Car

# Importing the libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch
import os

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the TD3 implementation
from td3 import TD3, ReplayBuffer, device

# Create images directory if it doesn't exist
if not os.path.exists("./images"):
    os.makedirs("./images")

# Check for required image files
required_images = {
    "citymap.png": "Required for displaying the map",
    "car.png": "Required for displaying the car",
    "MASK1.png": "Required for sensor visualization",
    "mask.png": "Required for sensor visualization",
    "sand.jpg": "Required for identifying the boundaries"
}

# Check if all required image files are present
for image_name, description in required_images.items():
    image_path = f"./images/{image_name}"
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Required image file not found: {image_name}\n"
            f"Description: {description}\n"
            f"Expected location: {image_path}\n"
            f"Please ensure all required image files are present in the ./images directory."
        )

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
# Initialize TD3 with state_dim=5 (3 signals + 2 orientations) and action_dim=1 (rotation)
brain = TD3(state_dim=5, action_dim=1, max_action=5.0)
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# Add episode tracking variables
total_timesteps = 0
episode_num = 1
episode_reward = 0
evaluation_rewards = []
EVALUATION_INTERVAL = 5000  # Evaluate every 5000 timesteps

# Initialize replay buffer for TD3
replay_buffer = ReplayBuffer(max_size=1e6)

# Define the three target points (A1, A2, A3) globally
target_points = [
    {'x': 1220, 'y': 618},  # A1
    {'x': 200, 'y': 285},   # A2
    {'x': 700, 'y': 265}    # A3
]

# Define the starting position for the car
start_position = {
    'x': 580,  # Starting x coordinate
    'y': 450   # Starting y coordinate
}

# textureMask = CoreImage(source="./kivytest/simplemask1.png")

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global current_target_index
    
    current_target_index = 0
    
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = target_points[0]['x']
    goal_y = target_points[0]['y']
    first_update = False
    global swap
    swap = 0

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the target marker class
class TargetMarker(Widget):
    def __init__(self, label_text, **kwargs):
        super(TargetMarker, self).__init__(**kwargs)
        with self.canvas:
            Color(0.5, 0, 0)  # Dark red color for initial state
            self.circle = Ellipse(size=(25, 25))  # Increased size for better visibility
            self.circle.pos = (self.x - 15, self.y - 15)  # Adjusted for new size
        
        # Add the label
        self.label = Label(
            text=label_text,
            color=(1, 1, 1, 1),  # White color
            font_size='20sp',  # Increased font size
            bold=True,
            size_hint=(None, None),
            size=(40, 30),  # Fixed size for the label
            center_x=self.center_x,
            center_y=self.center_y,
            padding=(5, 5)  # Add padding around text
        )
        self.add_widget(self.label)

    def update_pos(self, x, y):
        self.x = x
        self.y = y
        self.circle.pos = (x - 15, y - 15)  # Adjusted for new circle size
        self.label.center_x = x
        self.label.center_y = y

# Creating the start marker class
class StartMarker(Widget):
    def __init__(self, **kwargs):
        super(StartMarker, self).__init__(**kwargs)
        with self.canvas:
            Color(0, 0, 1)  # Blue color for start position
            self.circle = Ellipse(size=(25, 25))  # Slightly larger than target markers
            self.circle.pos = (self.x - 12.5, self.y - 12.5)  # Center the circle
            # Add a border
            Color(1, 1, 1)  # White border
            self.border = Ellipse(size=(30, 30))
            self.border.pos = (self.x - 15, self.y - 15)
        
        # Add the 'S' label
        self.label = Label(
            text='S',
            color=(0, 0, 1, 1),  # Blue color
            font_size='16sp',
            bold=True,
            center_x=self.center_x,
            center_y=self.center_y
        )
        self.add_widget(self.label)

    def update_pos(self, x, y):
        self.x = x
        self.y = y
        self.circle.pos = (x - 12.5, y - 12.5)
        self.border.pos = (x - 15, y - 15)
        self.label.center_x = x
        self.label.center_y = y

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    target_markers = []  # List to store target markers
    start_marker = None  # Start position marker
    position_history = []  # Track car's position history
    max_history_length = 50  # Maximum number of positions to track
    last_positions = set()  # Set of recent positions to detect loops
    
    def serve_car(self):
        print(f"Initializing car at position: ({start_position['x']}, {start_position['y']})")
        # Set the car's initial position to the defined start position
        self.car.x = start_position['x']
        self.car.y = start_position['y']
        print(f"Car position after setting: ({self.car.x}, {self.car.y})")
        self.car.velocity = Vector(6, 0)
        
        # Initialize start marker
        if self.start_marker is None:
            self.start_marker = StartMarker()
            self.start_marker.update_pos(start_position['x'], start_position['y'])
            self.add_widget(self.start_marker)
            print(f"Start marker initialized at: ({start_position['x']}, {start_position['y']})")
        
        # Initialize target markers
        self.target_markers = []
        for i, point in enumerate(target_points):
            marker = TargetMarker(label_text=f'A{i+1}')  # Create marker with label A1, A2, or A3
            marker.update_pos(point['x'], point['y'])
            self.add_widget(marker)
            self.target_markers.append(marker)
            print(f"Target marker {i+1} initialized at: ({point['x']}, {point['y']})")

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global target_points
        global current_target_index
        global replay_buffer
        global total_timesteps
        global episode_num
        global episode_reward
        global evaluation_rewards

        longueur = self.width
        largeur = self.height
        if first_update:
            print(f"First update - Window dimensions: {longueur}x{largeur}")
            init()

        # Add position logging every 100 frames
        if int(time.time() * 60) % 100 == 0:
            print(f"Current car position: ({self.car.x}, {self.car.y})")

        # Get current state
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        
        # Select action using TD3
        action = brain.select_action(np.array(state))
        # Convert continuous action to discrete rotation
        rotation_idx = int(np.clip((action[0] + 5) / 5, 0, 2))  # Map [-5,5] to [0,2]
        rotation = action2rotation[rotation_idx]
        
        # Store the old position for reward calculation
        old_x, old_y = self.car.x, self.car.y
        
        # Move the car
        self.car.move(rotation)
        
        # Update position history
        current_pos = (int(self.car.x), int(self.car.y))
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Update last positions set
        self.last_positions.add(current_pos)
        if len(self.last_positions) > 100:  # Keep last 100 positions
            self.last_positions.remove(next(iter(self.last_positions)))

        # Stronger penalty for revisiting recent positions (detect loops)
        if len(self.position_history) > 10:
            recent_positions = set(self.position_history[-10:])
            if current_pos in recent_positions:
                last_reward -= 2.0  # Increased penalty for revisiting positions

        # Penalty for maintaining similar angle (circular movement)
        if len(self.position_history) > 5:
            # Calculate angle changes over last 5 positions
            angle_changes = []
            for i in range(len(self.position_history)-1):
                dx = self.position_history[i+1][0] - self.position_history[i][0]
                dy = self.position_history[i+1][1] - self.position_history[i][1]
                angle = np.arctan2(dy, dx)
                angle_changes.append(angle)
            
            # If angles are very similar, likely moving in a circle
            if len(angle_changes) > 1:
                angle_variance = np.var(angle_changes)
                if angle_variance < 0.1:  # Small variance indicates similar angles
                    last_reward -= 1.5  # Penalty for maintaining similar angle

        # Enhanced circular movement detection
        if len(self.position_history) > 20:
            # Calculate the center of recent positions
            recent_x = sum(p[0] for p in self.position_history[-20:]) / 20
            recent_y = sum(p[1] for p in self.position_history[-20:]) / 20
            
            # Calculate average distance from center and variance
            distances = [np.sqrt((p[0] - recent_x)**2 + (p[1] - recent_y)**2) 
                        for p in self.position_history[-20:]]
            avg_dist = sum(distances) / 20
            dist_variance = np.var(distances)
            
            # If average distance is small and variance is low, likely moving in a tight circle
            if avg_dist < 40 and dist_variance < 100:
                last_reward -= 3.0  # Strong penalty for tight circular movement
            
            # Additional penalty for small radius turns
            if avg_dist < 30:
                last_reward -= 2.0  # Extra penalty for very tight circles

        # Get new state
        new_xx = goal_x - self.car.x
        new_yy = goal_y - self.car.y
        new_orientation = Vector(*self.car.velocity).angle((new_xx,new_yy))/180.
        new_state = [self.car.signal1, self.car.signal2, self.car.signal3, new_orientation, -new_orientation]
        
        # Calculate distance to target
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        # Update ball positions
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # Update target markers colors based on current target
        for i, marker in enumerate(self.target_markers):
            with marker.canvas:
                marker.canvas.clear()
                if i == current_target_index:
                    Color(0, 1, 0)  # Green for current target
                else:
                    Color(0.5, 0, 0)  # Dark red for other targets
                marker.circle = Ellipse(size=(25, 25))  # Match the new size
                marker.circle.pos = (marker.x - 15, marker.y - 15)  # Adjusted for new size

        # Reset position history when reaching a target to avoid false penalties
        if distance < 25:
            self.position_history = []
            self.last_positions = set()
            # Switch to next target point
            current_target_index = (current_target_index + 1) % len(target_points)
            goal_x = target_points[current_target_index]['x']
            goal_y = target_points[current_target_index]['y']
            last_reward = 20  # Reward for reaching target
            print(f"Switching to target point {current_target_index + 1}")

        # Calculate reward
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1

            # Calculate progress towards target
            progress = last_distance - distance
            
            # Stronger reward for making progress towards target
            if progress > 0:
                last_reward = 2.0  # Significantly increased reward for moving towards target
            else:
                # Stronger penalty for moving away from target
                last_reward = -1.0
            
            # Calculate angle to target
            target_angle = Vector(*self.car.velocity).angle((goal_x - self.car.x, goal_y - self.car.y))/180.
            angle_diff = abs(target_angle)
            
            # Stronger penalty for large angle differences from target
            if angle_diff > 0.5:  # If angle difference is more than 90 degrees
                last_reward -= angle_diff * 1.0  # Doubled penalty
            
            # Stronger penalty for sharp turns
            if abs(rotation) > 3:
                last_reward -= 1.0  # Doubled penalty for sharp turns
            
            # Bonus for maintaining good speed and moving in the right direction
            if abs(self.car.velocity[0]) > 1.5 and angle_diff < 0.3:
                last_reward += 0.5  # Increased bonus for good speed and direction
            
            # Stronger penalty for very slow movement
            if abs(self.car.velocity[0]) < 0.5:
                last_reward -= 0.5

        # Calculate distance to boundaries
        dist_to_left = self.car.x
        dist_to_right = self.width - self.car.x
        dist_to_top = self.height - self.car.y
        dist_to_bottom = self.car.y
        
        # Define boundary thresholds
        warning_threshold = 50  # Distance at which to start applying progressive penalties
        collision_threshold = 10  # Distance at which to apply collision penalties
        
        # Apply progressive penalties as the car gets closer to boundaries
        if dist_to_left < warning_threshold:
            last_reward -= (warning_threshold - dist_to_left) / warning_threshold
        if dist_to_right < warning_threshold:
            last_reward -= (warning_threshold - dist_to_right) / warning_threshold
        if dist_to_top < warning_threshold:
            last_reward -= (warning_threshold - dist_to_top) / warning_threshold
        if dist_to_bottom < warning_threshold:
            last_reward -= (warning_threshold - dist_to_bottom) / warning_threshold

        # Apply collision penalties and reset position
        if dist_to_left < collision_threshold:
            self.car.x = collision_threshold
            last_reward = -2  # Increased penalty for collision
            # Rotate car by 90 degrees to help it escape the boundary
            self.car.angle = (self.car.angle + 90) % 360
            # Reset velocity in the new direction
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        if dist_to_right < collision_threshold:
            self.car.x = self.width - collision_threshold
            last_reward = -2
            # Rotate car by -90 degrees to help it escape the boundary
            self.car.angle = (self.car.angle - 90) % 360
            # Reset velocity in the new direction
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        if dist_to_top < collision_threshold:
            self.car.y = self.height - collision_threshold
            last_reward = -2
            # Rotate car by 180 degrees to help it escape the boundary
            self.car.angle = (self.car.angle + 180) % 360
            # Reset velocity in the new direction
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
        if dist_to_bottom < collision_threshold:
            self.car.y = collision_threshold
            last_reward = -2
            # Rotate car by 180 degrees to help it escape the boundary
            self.car.angle = (self.car.angle + 180) % 360
            # Reset velocity in the new direction
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)

        # Add transition to replay buffer
        done = False  # In this continuous task, episodes don't really end
        replay_buffer.add((
            np.array(state, dtype=np.float32), 
            np.array(new_state, dtype=np.float32), 
            np.array([action], dtype=np.float32), 
            np.array([last_reward], dtype=np.float32), 
            np.array([done], dtype=np.float32)
        ))
        
        # Train TD3 if we have enough samples
        if len(replay_buffer.storage) > 100:  # Start training after 100 samples
            brain.train(replay_buffer, iterations=1)  # Train for 1 iteration per step
            
        # Update scores and last distance
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
            action_tensor = torch.FloatTensor(np.array([action]).reshape(1, -1)).to(device)
            scores.append(brain.critic.Q1(state_tensor, action_tensor).mean().item())
            
        last_distance = distance

        # Update episode tracking
        total_timesteps += 1
        episode_reward += last_reward

        # Print episode information at regular intervals
        if total_timesteps % 1000 == 0:
            print(f"\nEpisode Summary:")
            print(f"Total Timesteps: {total_timesteps}")
            print(f"Episode Num: {episode_num}")
            print(f"Episode Reward: {episode_reward:.4f}")
            print(f"Current Distance to Target: {distance:.2f}")
            print(f"Current Speed: {np.linalg.norm(self.car.velocity):.2f}")
            episode_reward = 0
            episode_num += 1

        # Evaluation step
        if total_timesteps % EVALUATION_INTERVAL == 0:
            avg_reward = sum(evaluation_rewards) / len(evaluation_rewards) if evaluation_rewards else 0
            print("\n" + "="*50)
            print("Evaluation Summary:")
            print(f"Total Timesteps: {total_timesteps}")
            print(f"Average Reward: {avg_reward:.6f}")
            print(f"Current Exploration Noise: {brain.exploration_noise:.4f}")
            print(f"Replay Buffer Size: {len(replay_buffer.storage)}")
            print("="*50 + "\n")
            evaluation_rewards = []
        else:
            evaluation_rewards.append(last_reward)

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        
        # Schedule car initialization after a short delay
        Clock.schedule_once(lambda dt: parent.serve_car(), 0.1)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("Saving brain...")
        brain.save("td3_model", "./checkpoints")
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("Loading last saved brain...")
        brain.load("td3_model", "./checkpoints")

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
