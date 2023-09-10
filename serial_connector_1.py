import serial
from time import sleep

class SerialConnector:
  def connect(self, port):
    try:
      # Defines the connection parameters for the serial connection
      self.ser = serial.Serial(port, 115200, timeout=1)
      # Checks to make sure that the name of the connected port is as expected
      if self.ser.name == port:
          self.connected = True
          print("Connected to", self.ser.name)
      self.ser.write(b'a')
    except:
      self.connected = False
      print("Failed to connect to serial port")

  # Reads all serial messages in the queue
  # Returns a list of all messages or None if there are no messages
  def read_serial_messages(self):
    # if there haven't been any messages, returns None
    if self.ser.in_waiting == 0 or self.connected == False:
      return None

    lines = []
    while self.ser.in_waiting > 0:
      lines.append(self.ser.readline().decode('utf-8').strip())

#    print(lines)
    return lines

  def get_orientation(self):
    lines = None
    while not lines:
      lines = self.read_serial_messages()
    data = lines[-1]
    if data[0] != '[' or data[-1] != ']':
      return (0, 0, 0)
    angles = data[1:-1].split(',')
    return angles



