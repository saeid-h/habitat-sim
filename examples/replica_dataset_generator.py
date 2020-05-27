# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import habitat_sim
from PIL import Image
import sintel_io as sintel
import cv2
import argparse
import os


def main():

	parser = argparse.ArgumentParser(description='Dataset generator example.')
	parser.add_argument('--dataset_path',       type=str,   help='Path to dataset', default='data/replica_dataset/')
	parser.add_argument('--save_path',          type=str,   help='Path to save', default='data/replica_reconstructed/')
	parser.add_argument('--forward_prob',       type=float, help='forward step probability', default=0.25)
	parser.add_argument('--left_prob',          type=float, help='left step probability', default=0.15)
	parser.add_argument('--focal',          	type=float, help='left step probability', default=300)
	parser.add_argument('--image_size',         type=int,   nargs=2,  help='generating image size.', default=[512, 512])
	parser.add_argument('--repeat_room',        type=int,   help='repeat the image generating process.', default=5)
	parser.add_argument('--up_pos',         	type=float, nargs=2,  help='random position limitis for sensor height.', default=[0.5, 2.0])
	parser.add_argument('--left_pos',         	type=float, nargs=2,  help='random position limitis for sensor left.', default=[-1.0, 1.0])
	parser.add_argument('--back_pos',         	type=float, nargs=2,  help='random position limitis for sensor back.', default=[-1.0, 1.0])
	
	args = parser.parse_args()
	replica_dataset_generator(args)


def setup_agent(dataset_path, random_pos, image_size, focal):

	random_up, random_left, random_back = random_pos
	backend_cfg = habitat_sim.SimulatorConfiguration()
	backend_cfg.scene.id = (dataset_path)
	
	hfov = 2*np.arctan(0.5*image_size[1]/focal)/np.pi*180

	# First, let's create a stereo RGB agent
	left_rgb_sensor = habitat_sim.SensorSpec()
	# Give it the uuid of left_sensor, this will also be how we
	# index the observations to retrieve the rendering from this sensor
	left_rgb_sensor.uuid = "left_rgb_sensor"
	left_rgb_sensor.resolution = image_size
	# The left RGB sensor will be 1.5 meters off the ground
	# and 0.25 meters to the left of the center of the agent
	left_rgb_sensor.position = random_up * habitat_sim.geo.UP + random_left * habitat_sim.geo.LEFT + random_back * habitat_sim.geo.BACK
	left_rgb_sensor.parameters["hfov"] = str(hfov)

	# Same deal with the right sensor
	right_rgb_sensor = habitat_sim.SensorSpec()
	right_rgb_sensor.uuid = "right_rgb_sensor"
	right_rgb_sensor.resolution = image_size
	# The right RGB sensor will be 1.5 meters off the ground
	# and 0.25 meters to the right of the center of the agent
	right_rgb_sensor.position = left_rgb_sensor.position + 0.5 * habitat_sim.geo.RIGHT
	right_rgb_sensor.parameters["hfov"] = str(hfov)

	# Now let's do the exact same thing but for a depth camera stereo pair!
	left_depth_sensor = habitat_sim.SensorSpec()
	left_depth_sensor.uuid = "left_depth_sensor"
	left_depth_sensor.resolution = image_size
	left_depth_sensor.position = left_rgb_sensor.position
	# The only difference is that we set the sensor type to DEPTH
	left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
	left_depth_sensor.parameters["hfov"] = str(hfov)

	right_depth_sensor = habitat_sim.SensorSpec()
	right_depth_sensor.uuid = "right_depth_sensor"
	right_depth_sensor.resolution = image_size
	right_depth_sensor.position = right_rgb_sensor.position
	# The only difference is that we set the sensor type to DEPTH
	right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
	right_depth_sensor.parameters["hfov"] = str(hfov)
	
	# Now we simly set the agent's list of sensor specs to be the two specs for our two sensors
	agent_config = habitat_sim.AgentConfiguration()
	agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor, left_depth_sensor, right_depth_sensor]

	sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
	
	return sim
	
	
def capture_scene(sim, save_path, room, forward_probability, left_probability, focal):
	seq = 0
	rotation = 0
	last_move = ''
	os.system('mkdir -p ' + save_path)
	os.system('mkdir -p ' + save_path + 'image_left/' + room)
	os.system('mkdir -p ' + save_path + 'image_right/'+ room)
	os.system('mkdir -p ' + save_path + 'depth_left/' + room)
	os.system('mkdir -p ' + save_path + 'depth_right/'+ room)
	while rotation < 36:
		# Just spin in a circle
		if np.random.uniform(0.0, 1.0) < forward_probability:
			obs = sim.step("move_forward")
			last_move = 'forward'
		elif last_move != 'right' and np.random.uniform(0.0, 1.0) < left_probability:
			obs = sim.step("turn_left")
			rotation -= 1
			last_move = 'left'
		elif last_move != 'left':
			obs = sim.step("turn_right")
			rotation += 1
			last_move = 'right'
		else:
			obs = sim.step("move_forward")
			last_move = 'forward'

		left_rgb_image = obs["left_rgb_sensor"]
		right_rgb_image = obs["right_rgb_sensor"]
		left_depth_image = obs["left_depth_sensor"]
		right_depth_image = obs["right_depth_sensor"]

		if not obs['collided']:
			print ('Saving {}/{}_{} ...'.format(room, str(int(focal)).zfill(4), str(seq).zfill(4)))
			file_name = '{}/{}/{}_{}_{}.{}'.format(save_path+'/image_left', room, str(int(focal)).zfill(4), str(seq).zfill(4), 'left', 'png')
			Image.fromarray(left_rgb_image[..., 0:3]).save(file_name)
			file_name = file_name.replace('_left', '_right')
			Image.fromarray(right_rgb_image[..., 0:3]).save(file_name)
			file_name = '{}/{}/{}_{}_{}.{}'.format(save_path+'/depth_left', room, str(int(focal)).zfill(4), str(seq).zfill(4), 'left', 'dpt')
			sintel.depth_write(file_name, left_depth_image)
			file_name = file_name.replace('_left', '_right')
			sintel.depth_write(file_name, right_depth_image)
		
			seq += 1
		
	sim.close()
	
	return seq
	
	
def replica_dataset_generator(args):	
	rooms = ['apartment_0', 'apartment_1', 'apartment_2', 
			'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_3', 'frl_apartment_4', 'frl_apartment_5', 
			'hotel_0', 
			'office_0', 'office_1', 'office_2', 'office_3', 'office_4', 
			'room_0', 'room_1', 'room_2', ]
	room = rooms[7]
	
	save_path = args.save_path
	image_size = args.image_size
	forward_probability = args.forward_prob
	left_probability = args.left_prob
	random_up = np.random.uniform(args.up_pos[0], args.up_pos[1])
	random_left = np.random.uniform(args.left_pos[0], args.left_pos[1])
	random_back = np.random.uniform(args.back_pos[0], args.back_pos[1])
	random_pos = [random_up, random_left, random_back]
	
	assert 0.0 <= forward_probability <= 1
	assert 0.0 <= left_probability <= 1
	assert (1-forward_probability) * left_probability < 0.5
	
	n = 500 / (args.repeat_room-1)
	focals = range(100,601, n) if args.focal is None else [args.focal]*args.repeat_room

	for room in rooms:
		dataset_path = args.dataset_path + room + '/habitat/mesh_semantic.ply'
		for rep in range(args.repeat_room):
			sim = setup_agent(dataset_path, random_pos, image_size, focals[rep])
			capture_scene(sim, save_path, room, forward_probability, left_probability, focals[rep])
	
	

		

if __name__ == "__main__":
	main()
