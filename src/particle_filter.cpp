/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 50;
  particles.resize(num_particles);
  weights.resize(num_particles, 1.0f);

  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);
  
  for (int i = 0; i < num_particles; i++) {
    Particle p{};
    p.id = i;
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen),
    p.weight = 1.0f;
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], double v, double dtheta) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for (int i = 0; i < particles.size(); i++) {
    auto p = particles[i];
    if (dtheta == 0) {
      p.x += v * cos(p.theta) * dt;
      p.y += v * sin(p.theta) * dt;
    } else {
      p.x += v * (sin(p.theta + dtheta * dt) - sin(p.theta)) / dtheta;
      p.y += v * (cos(p.theta) - cos(p.theta + dtheta * dt)) / dtheta;
    }
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dtheta * dt + dist_theta(gen);
    particles[i] = p;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  for (int i = 0; i < particles.size(); i++) {
    auto p = particles[i];
  
    // reset weight, sense_x, sense_y and associations
    p.weight = 1.0;
    p.sense_x.clear();
    p.sense_y.clear();
    p.associations.clear();
    p.sense_x.reserve(observations.size());
    p.sense_y.reserve(observations.size());
    p.associations.reserve(observations.size());
    
    for (auto obs : observations) {
      double obs_map_x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
      double obs_map_y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
      
      float min_dist = -1;
      
      auto landmarks = map_landmarks.landmark_list;
      Map::single_landmark_s best_landmark;
      
      for (auto landmark : landmarks) {
        if (fabs(dist(landmark.x_f, landmark.y_f, p.x, p.y)) > sensor_range) {
          continue;
        }
        
        float euclidean_dist = fabs(dist(landmark.x_f, landmark.y_f, obs_map_x, obs_map_y));
        if (min_dist < 0 || euclidean_dist < min_dist) {
          min_dist = euclidean_dist;
          best_landmark = landmark;
        }
      }
      
      if (min_dist < 0) {
        continue;
      }
      p.sense_x.push_back(obs_map_x);
      p.sense_y.push_back(obs_map_y);
      p.associations.push_back(best_landmark.id_i);
      
      // compute observation partial probability for the selected landmark
      float sigma_x = std_landmark[0];
      float sigma_y = std_landmark[1];
      float exponent_x = pow(obs_map_x - best_landmark.x_f, 2) / 2 * sigma_x * sigma_x;
      float exponent_y = pow(obs_map_y - best_landmark.y_f, 2) / 2 * sigma_y * sigma_y;
      p.weight *= exp(-(exponent_x + exponent_y)) / 2 * M_PI * sigma_x * sigma_y;
    }
    particles[i] = p;
    weights[i] = p.weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> p_resample;
  
  for (int i = 0; i < num_particles; i++) {
    p_resample.push_back(particles[d(gen)]);
  }
  particles = p_resample;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
