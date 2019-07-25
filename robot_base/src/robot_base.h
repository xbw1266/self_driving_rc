#include "ros/ros.h"
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>


namespace robot_base{

class Robot_base{
    public:

    Robot_base();

    ~Robot_base();


    private:

    ros::Subscriber right_vel_sub_;
    ros::Subscriber left_vel_sub_;

    //pubilisher for odometry information
    ros::Publisher Odom_pub_;
    //base odometry tf
    geometry_msgs::TransformStapmed odom_tf_;
    //base odometry tf broadcaster
    tf::TransformBroadcaster tf_broadcaster_;
    //odometry message
    nav_msgs::Odometry odom_;



};
}
