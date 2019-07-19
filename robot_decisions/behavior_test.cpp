#include <ros/ros.h>
#include "blackboard/blackboard.h"
#include "behavior_tree/behavior_tree.h"

void Command();
char command = '0';

int main(int argc, char **argv) {
  ros::init(argc, argv, "behavior_test_node");

  ROS_ERROR("do something");

  return 0;
}

