/****************************************************************************
 *  Copyright (C) 2019 RoboMaster.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of 
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/
#ifndef ROBORTS_DECISION_BLACKBOARD_H
#define ROBORTS_DECISION_BLACKBOARD_H

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <robot_decisions/behavior_msg.h>

namespace robot_decisions{

class Blackboard {
 public:
  typedef std::shared_ptr<Blackboard> Ptr;

  explicit Blackboard(){
      behavior_sub_ = nh_.subscribe<behavior_msg>("behavior_msg",1,&Blackboard::RobotBehaviorCallback,this);
  }

  ~Blackboard() = default;

  void RobotBehaviorCallback(const robot_decisions::behavior_msg::ConstPtr& info){
      stop_sign_ = info->stop_sign;
      traffic_light_ = info->traffic_light;
      move_ = info->move;
  }

  const bool GetStopSign(){
    return stop_sign_;
  }

  bool IsStopSign(){
    return GetStopSign();
  }

  const bool GetTrafficLight(){
    return traffic_light_;
  }

  bool IsRedLight(){
    return GetTrafficLight();
  }
  
  const std::string GetMove(){
    return move_;
  }

 private:
 
  //behavior subscribe
  ros::Subscriber behavior_sub_;

  //node handler
  ros::NodeHandle nh_;

  //behavior
  std::string move_;
  bool stop_sign_;
  bool traffic_light_;

};
} //namespace roborts_decision
#endif //ROBORTS_DECISION_BLACKBOARD_H
