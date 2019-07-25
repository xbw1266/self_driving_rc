#include <ros/ros.h>
#include "blackboard/blackboard.h"
#include "behavior_tree/behavior_tree.h"
#include "Actions/Stop_action.h"
#include "Actions/Redlight_action.h"
#include "Actions/Move_action.h"

void Command();
char command = '0';

int main(int argc, char **argv) {
  ros::init(argc, argv, "behavior_test_node");

  auto blackboard_ptr = std::make_shared<robot_decisions::Blackboard>();

  /************selections**************/
  auto is_stopsign = std::make_shared<robot_decisions::SelectorNode>(
          "is_stopsign_seletor",
          blackboard_ptr);

  auto is_redlight = std::make_shared<robot_decisions::SelectorNode>(
          "is_redlight_seletor",
          blackboard_ptr);

  auto start_sel = std::make_shared<robot_decisions::SelectorNode>(
          "start_selector",
          blackboard_ptr);
  /************pre condition***********/
  auto is_stopsign_condition = std::make_shared<robot_decisions::PreconditionNode>(
          "is_stopsign_condition",
          blackboard_ptr,
          [&](){
            return blackboard_ptr->IsStopSign();
          },
          robot_decisions::AbortType::SELF);

  auto is_redlight_condition = std::make_shared<robot_decisions::PreconditionNode>(
          "is_redlight_condition",
          blackboard_ptr,
          [&](){
            return blackboard_ptr->IsRedLight();
          },
          robot_decisions::AbortType::SELF);
  /***********actions******************/
  auto stop_action = std::make_shared<robot_decisions::StopAction>(blackboard_ptr);
  auto redlight_action = std::make_shared<robot_decisions::RedlightAction>(blackboard_ptr);
  auto move_action = std::make_shared<robot_decisions::MoveAction>(blackboard_ptr);


  /************build tree**************/
  is_stopsign_condition->SetChild(is_stopsign);
  is_stopsign->AddChildren(stop_action);

  is_redlight_condition->SetChild(is_redlight);
  is_redlight->AddChildren(redlight_action);
  /************root root:start_sel*****/
  start_sel->AddChildren(is_stopsign_condition);
  start_sel->AddChildren(is_redlight_condition);
  start_sel->AddChildren(move_action);

  robot_decisions::BehaviorTree root(start_sel,1000); //int is excution duration in ms

  root.Run();


  return 0;
}

