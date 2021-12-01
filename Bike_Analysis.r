library(tidyverse)

library(lubridate)

library(ggplot2)

q1<-read_csv("C:\\Users\\26097\\Documents\\Divvy_Trips_2020_Q1.csv")

m4<-read_csv("C:\\Users\\26097\\Documents\\202004-divvy-tripdata.csv")

m5<-read_csv("C:\\Users\\26097\\Documents\\202005-divvy-tripdata.csv")

m6<-read_csv("C:\\Users\\26097\\Documents\\202006-divvy-tripdata.csv")

m7<-read_csv("C:\\Users\\26097\\Documents\\202007-divvy-tripdata.csv")

m8<-read_csv("C:\\Users\\26097\\Documents\\202008-divvy-tripdata.csv")

m9<-read_csv("C:\\Users\\26097\\Documents\\202009-divvy-tripdata.csv")

m10<-read_csv("C:\\Users\\26097\\Documents\\202010-divvy-tripdata.csv")

m11<-read_csv("C:\\Users\\26097\\Documents\\202011-divvy-tripdata.csv")

m12<-read_csv("C:\\Users\\26097\\Documents\\202012-divvy-tripdata.csv")

str(q1)
str(m4)
str(m5)
str(m6)
str(m7)
str(m8)
str(m9)
str(m10)
str(m11)
str(m12)


class(q1$start_station_id)
class(m4$start_station_id)
class(m5$start_station_id)
class(m6$start_station_id)
class(m7$start_station_id)
class(m8$start_station_id)
class(m9$start_station_id)
class(m10$start_station_id)
class(m11$start_station_id)
class(m12$start_station_id)

m12$start_station_id <-as.numeric(m12$start_station_id)
class(m12$start_station_id)

class(q1$end_station_id)
class(m4$end_station_id)
class(m5$end_station_id)
class(m6$end_station_id)
class(m7$end_station_id)
class(m8$end_station_id)
class(m9$end_station_id)
class(m10$end_station_id)
class(m11$end_station_id)
class(m12$end_station_id)

m12$end_station_id <-as.numeric(m12$end_station_id)
class(m12$end_station_id)

df<-rbind(q1, m4, m5, m6, m7, m8, m9, m10, m11,m12)
head(df)
dim(df)

dim(df)
summary (df)

df <- df%>%select(-c(start_lat,start_lng, end_lat, end_lng))

df$date <-as.Date(df$started_at)
df$month <-format(as.Date(df$date), "%m")
df$year <-format(as.Date(df$date), "%Y")
df$day_of_week <-format(as.Date(df$date), "%A")
df$duration <- difftime(df$ended_at,df$started_at)
df$duration <-as.numeric(df$duration)
is.numeric(df$duration)

df1 <-df[!(df$start_station_name == "HQ QR"|df$duration<0),]
mean(df1$duration)
nas<- apply(df1, 1, function(x){any(is.na(x))})
sum(nas)

df2 <-df1[!nas,]
nrow(df2)
nrow(df1)
summary(df2$duration)

summary(df2$duration)
aggregate(df2$duration~df2$member_casual, FUN=mean)
aggregate(df2$duration~df2$member_casual, FUN=median)
aggregate(df2$duration~df2$member_casual, FUN=max)


df2$day_of_week <- ordered(df2$day_of_week, levels=c("Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
aggregate(df2$duration~df2$member_casual+df2$day_of_week, FUN=median)


 df2$month_name <- ordered(df2$month_name, levels=c("Jan","Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug","Sep", "Oct", "Nov", "Dec"))
aggregate(df2$duration~df2$member_casual+df2$month_name, FUN=median)

df2$time <-strftime(df2$started_at, format="%H")
head(df2$time)

aggregate(df2$duration~df2$member_casual+df2$time, FUN=median)

df2 %>%
 mutate(weekday=wday(started_at, label=TRUE))%>%
 group_by(member_casual, weekday) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, weekday)

df2 %>%
 mutate(weekday=wday(started_at, label=TRUE))%>%
 group_by(member_casual, weekday) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, weekday)

require(scales)

df2 %>%
 mutate(weekday=wday(started_at, label=TRUE))%>%
 group_by(member_casual, weekday) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, weekday) %>%
 ggplot(aes(x=weekday, y=number_of_rides, fill=member_casual))+
 geom_col(position="dodge")+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 scale_y_continuous(labels = comma)

df2 %>%
 mutate(weekday=wday(started_at, label=TRUE))%>%
 group_by(member_casual, weekday) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, weekday) %>%
 ggplot(aes(x=weekday, y=avg_duration, fill=member_casual))+
 geom_col(position="dodge")+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 scale_y_continuous(labels = comma)

 df2 %>%
 group_by(member_casual, month_name) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, month_name) %>%
 ggplot(aes(x=month_name, y=avg_duration, fill=member_casual))+
 geom_col(position="dodge")+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 scale_y_continuous(labels = comma)

df2 %>%
 group_by(member_casual, month_name) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, month_name) %>%
 ggplot(aes(x=month_name, y=number_of_rides, fill=member_casual))+
 geom_col(position="dodge")+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 scale_y_continuous(labels = comma)

df2 %>%
 group_by(member_casual, time) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, time) %>%
 ggplot(aes(x=time, y=avg_duration, fill=member_casual))+
 geom_col(position="dodge")+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 scale_y_continuous(labels = comma)

df2 %>%
 group_by(member_casual, time) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, time) %>%
 ggplot(aes(x=time, y=number_of_rides, fill=member_casual))+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 geom_dotplot(binaxis = "y", stackdir = "center")

df2 %>%
 group_by(member_casual, time) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, time) %>%
 ggplot(aes(x=time, y=number_of_rides, fill=member_casual))+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 geom_dotplot(binaxis = "y", stackdir = "center")

install.packages("Hmisc")
load.packages(Hmisc)

df2 %>%
 group_by(member_casual, time) %>%
 summarise(number_of_rides=n(), avg_duration=mean(duration)) %>%
 arrange(member_casual, time) %>%
 ggplot(aes(x=time, y=number_of_rides, fill=member_casual))+
 geom_bar(stat="identity",position=position_dodge())+
 theme_minimal()+
 scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
 scale_y_continuous(labels = comma)


