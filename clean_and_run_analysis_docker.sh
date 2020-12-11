docker run --rm -e PASSWORD="test" -v "/$(pwd)":/home/seismophobia/ dbandrews/seismophobia:v0.4.0 make directory=/home/seismophobia clean all
