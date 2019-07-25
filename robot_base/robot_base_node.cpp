#include "src/robot_base.h"


int main(int argc, char ** argv){
    ros::init(argc,argv,"robot_base_node");
    ros::NodeHandle nh;

    if (ros::ok())
        return 0;
    return 0;
}
