# Method Overview

## Concept

The Congruity Global Instability Engine is designed to detect and interpret systemic deviations in seismic activity, not as isolated events but as evolving dynamic phases.

Instead of focusing on single earthquakes, the system analyzes how activity changes over time, space, and intensity, identifying patterns of activation, persistence, and transition.

The key idea is:

**instability is not an event, it is a process**

## Public radar layer

The public radar included in this repository represents a simplified surface layer of the system.

It transforms raw seismic data into:

, spatial aggregation  
, density and energy estimation  
, deviation from local equilibrium  
, clustering patterns  
, dynamic instability phases  

This layer is intentionally simplified and designed for visualization and exploratory analysis.

## Internal engine logic (abstracted)

At a deeper level, the full system operates as a multi layer temporal engine.

The internal logic can be summarized as:

### 1. Signal extraction
Daily aggregation of seismic activity:
, event count  
, magnitude distribution  
, spatial dispersion  
, energy proxy  

### 2. Memory accumulation
The system builds a temporal memory of activity:
, persistence of activity  
, cumulative deviation  
, recent vs past balance  

This allows distinguishing:
, noise  
, temporary spikes  
, sustained activation  

### 3. Temporal coherence
The system evaluates whether changes are:
, isolated  
, sustained  
, accelerating  

This is done through:
, consecutive increases  
, multi day consistency  
, suppression of single day spikes  

### 4. Phase classification
Activity is interpreted as a phase:

, quiet  
, pressure  
, early activation  
, confirmed activation  
, strong activation  

These phases are not fixed thresholds, but dynamic states emerging from the system behavior.

### 5. Decision layer (not public)
A higher level policy determines when a signal becomes actionable:

, filtering of noise  
, suppression of unstable spikes  
, detection of meaningful transitions  
, alert emission logic  

This layer is not fully exposed in the public repository.

## Interpretation

The system does not attempt to predict exact earthquake events.

Instead, it identifies:

, activation windows  
, shifts in seismic regime  
, phases of elevated instability  

This makes it closer to a **dynamic system observer** than a traditional predictor.

## Key principle

The most important architectural concept is:

**the model observes, the policy decides**

The radar shows what is happening.

The internal engine determines what is meaningful.

## Limitations

The public version has several limitations:

, simplified feature set  
, no full temporal decision engine  
, no full validation layer  
, no access to private research modules  

It should be interpreted as a **visual and conceptual interface**, not as a complete system.

## Positioning

This project should be understood as:

, an experimental instability observatory  
, a spatial visualization layer  
, a conceptual bridge between raw seismic data and systemic interpretation  

It is not a complete prediction system, but a foundation for one.
