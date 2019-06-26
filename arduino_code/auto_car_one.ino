
#include <SoftwareSerial.h>
#define mtrA_drive_1 4
#define mtrA_drive_2 7
#define mtrA_pwm     5

int steer = 90;
#include <Servo.h>

Servo myservo; 

SoftwareSerial mySerial(10, 11); // RX, TX


//uint8_t incomingByte;

void setup() {
  Serial.begin(9600);
  pinMode(mtrA_drive_1, OUTPUT);
  pinMode(mtrA_drive_2, OUTPUT);
  pinMode(mtrA_pwm, OUTPUT);
  pinMode(13,OUTPUT);
  myservo.attach(9);
  ///mySerial.begin(9600);
  Serial.println("Arduino ready");
  //myservo.write(steer);   
  delay(100);

//  myservo.write(60);   
//  delay(2000);
//  myservo.write(120);   
//  delay(2000);
  
  //straight(0.5, 1);
  //delay(4000);
//  straight(0.2, 0);
//  delay(4000);
//  straight(0.8, 1);
//  delay(4000);
mtrA_brake();

}

void loop() {
  if (Serial.available() > 0) 
  {
      // read the incoming byte:
      char incomingByte;
      incomingByte = Serial.read();
                //Serial.print("its\t");
      Serial.println(incomingByte);
                //Serial.flush();
      if(incomingByte=='S')
      {mtrA_brake();
      digitalWrite(13,!digitalRead(13));}
      else if (incomingByte=='F')straight(0.5, 1);
      else if (incomingByte=='B')straight(0.4, 0);
      else if (incomingByte=='L')steering_L();
      else if (incomingByte=='R')steering_R();
      else {mtrA_brake();}
//   switch (incomingByte)
//   {
//    case 67:
//      mtrA_brake();
//      digitalWrite(13,!digitalRead(13));
//    case 85:
//      straight(0.5, 1);
//    }
      }
//  if (incomingByte == "C"){mtrA_brake();}
//  else if (incomingByte == "U"){straight(0.5, 1);}
  delay(5);

}


// 
void straight(float main_speed , uint8_t  F)
{
  analogWrite(mtrA_pwm, uint8_t(main_speed*255));
  if(main_speed!=0){
    if(F==1)set_forward();
    else if(F==0) set_backward();
    }
  else {mtrA_brake();}
  }


void set_forward()
{ digitalWrite(mtrA_drive_1,1);digitalWrite(mtrA_drive_2,0);
  }
void set_backward()
{ digitalWrite(mtrA_drive_1,0);digitalWrite(mtrA_drive_2,1);
  }
void mtrA_brake()
{ digitalWrite(mtrA_drive_1,0);digitalWrite(mtrA_drive_2,0);analogWrite(mtrA_pwm, 0);
  }

void steering_L()
{steer -= 3;   myservo.write(steer);   delay(10);}
void steering_R()
{steer += 3;   myservo.write(steer);   delay(10);}
