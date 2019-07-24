/****************************************************************************
 *  Copyright (C) 2019 Linjian Xiang.
 *
 *Redlight action
 * 
 ***************************************************************************/

#include "../blackboard/blackboard.h"
#include "../behavior_tree/behavior_state.h"
#include <std_msgs/String.h>
#include <ros/ros.h>
namespace robot_decisions {
class MoveAction : public ActionNode {
 public:
  MoveAction(Blackboard::Ptr &blackboard) : ActionNode::ActionNode("move_action", blackboard)
  {
	keys_pub_ = nh_.advertise<std_msgs::String>("/keys",1,true);
	
  }

  virtual ~MoveAction() = default;

 private:

  virtual void OnInitialize() {
    ROS_INFO("%s %s\n", name_.c_str(), __FUNCTION__);
  }

  virtual void OnTerminate(BehaviorState state) {
	std_msgs::String msg;
	std::stringstream ss;
    switch (state){
      case BehaviorState::IDLE:
          ROS_INFO("%s %s IDLE!\n", name_.c_str(), __FUNCTION__);
          break;
      case BehaviorState::SUCCESS:
	  //ss << "S";
	  //msg.data = ss.str();
	  keys_pub_.publish(blackboard_ptr_->GetMove());
	  ROS_ERROR("move \n");
          ROS_INFO("%s %s SUCCESS!\n", name_.c_str(), __FUNCTION__);
          break;
      case BehaviorState::FAILURE:
          ROS_INFO("%s %s FAILURE!\n", name_.c_str(), __FUNCTION__);
          break;
      default:
          ROS_INFO("%s %s ERROR!\n", name_.c_str(), __FUNCTION__);
          return;
    }
  }

  virtual BehaviorState Update() {

    return BehaviorState::SUCCESS;
  }

 private:
  //! executor
  ros::NodeHandle nh_;
  ros::Publisher keys_pub_;
  bool wait_for_supply_;
};
}

