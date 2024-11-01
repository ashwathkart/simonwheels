import rospy
from novatel_gps_msgs.msg import Inspva  # Adjusted to match the correct message type

def gps_callback(msg):
    # Print the parsed data
    print("Latitude:", msg.latitude)
    print("Longitude:", msg.longitude)
    print("Height:", msg.height)
    print("Roll:", msg.roll)
    print("Pitch:", msg.pitch)
    print("Azimuth:", msg.azimuth)
    # Use msg.header.stamp if timestamp from GPS data is required
    print("Timestamp:", msg.header.stamp.to_sec())

def main():
    rospy.init_node('gps_data_parser')
    rospy.Subscriber('/novatel/inspva', Inspva, gps_callback)  # Remove trailing slash
    rospy.spin()

if __name__ == '__main__':
    main()