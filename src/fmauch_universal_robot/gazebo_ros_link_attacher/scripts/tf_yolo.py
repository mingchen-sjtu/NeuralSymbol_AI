import rospy
import tf2_ros
import tf
# from tf2_geometry_msgs import PoseStamped,PointStamped
from geometry_msgs.msg import PointStamped
 
if __name__ == "__main__":
    
    rospy.init_node('tf_yolo')
    
    
    # tf_buffer=tf2_ros.Buffer()
    # tf_listener=tf2_ros.TransformListener(tf_buffer)
    tf_listener=tf.TransformListener()
    
    
    p=PointStamped()
 
    # p.pose.position.x=1
    # p.pose.position.y=1
    # p.pose.position.z=2
    # p.pose.orientation.x=0
    # p.pose.orientation.y=0
    # p.pose.orientation.z=0
    # p.pose.orientation.w=0

    tf_listener.waitForTransform( 'world', 'M6_bolt_1/base_link',rospy.Time(), rospy.Duration(2))

    p.header.stamp=rospy.Time()
    p.header.frame_id='M6_bolt_1/base_link'
    p.point.x=1
    p.point.y=1
    p.point.z=2
    

    p_tgt = tf_listener.transformPoint('world', p)
 
    
   
 
    # res=tf_buffer.transform(p,'world',timeout=rospy.Duration(5))
    
    
    print (p_tgt)