import socket               # Import socket module
import subprocess

s = socket.socket()
host = socket.gethostname()# Create a socket object
s.bind((host, 0))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
print (':'.join(str(e) for e in s.getsockname()))
FD = "/home/weiyang/下载/eplus_RL/eplus_env_v1/eplus_env"
weather_path = FD + "/envs/weather/pittsburgh_TMY3.epw"
idf_path= FD + '/envs/eplus_models/rl_exp_part_1/idf/1.csl.vavDx.light.pittsburgh.idf.env'
eplus_working_dir = "."
# eplus_process = subprocess.Popen('%s -w %s -d %s %s'
# 								 % ("/home/weiyang/下载/eplus_RL/eplus_env_v1/eplus_env/envs/EnergyPlus-8-3-0" + '/energyplus', weather_path,
# 									"./output", idf_path),
# 								 shell=True,
# 								 cwd=eplus_working_dir,
# 								 stdout=subprocess.PIPE,
# 								 stderr=subprocess.PIPE,
# 								 )
while True:
    c, addr = s.accept()     # Establish connection with client.
    print('Got connection from', addr)
        # while True:
    recv = c.recv(1024).decode()
    print(recv)
	# 	if recv == 'get':
	# 		c.sendall(bytearray(str([1,2,3,4]), encoding = 'utf-8'));
	# 	else:
	# 		c.sendall(b'I don not understand')
