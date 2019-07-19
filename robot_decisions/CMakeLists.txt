project(robot_decisions)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_module)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  nav_msgs
  geometry_msgs
  message_generation
  std_msgs
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#generate proto files

add_message_files(FILES
    behavior_msg.msg
    )

generate_messages(DEPENDENCIES
    std_msgs
    )

catkin_package(
    CATKIN_DEPENDS message_runtime
    )

add_executable(behavior_test_node
  behavior_test.cpp
  blackboard/blackboard.h
  behavior_tree/behavior_node.h
  )

target_link_libraries(behavior_test_node
  PRIVATE
  ${catkin_LIBRARIES}
  )


