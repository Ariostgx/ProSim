from enum import IntEnum
from collections import defaultdict

class V_Action_MotionTag(IntEnum):
  Stopping = 0
  Accelerate = 1
  Decelerate = 2
  KeepSpeed = 3
  LeftLaneChange = 4
  RightLaneChange = 5
  KeepLane = 6
  LeftTurn = 7
  RightTurn = 8
  Straight = 9
  Parked = 10

class V2V_MotionTag(IntEnum):
  Following = 0
  ParallelDriving = 1
  Merging = 2
  ByPassing = 3
  Overtaking = 4

class MotionTags:
    def __init__(self, motion_tags):
        # List[Dict]
        self.motion_tags = motion_tags
    
    def __to__(self, device, non_blocking=False):
        return self

    def __collate__(self, batch):
        result = []
        for item in batch:
            result += item.motion_tags

        return MotionTags(result)

    def __getitem__(self, idx):
        return self.motion_tags[idx]

    def __len__(self):
        return len(self.motion_tags[0])

def integrate_motion_tags(snapshot_tags, tolerance=5):
    integrated_tags = {}

    # Iterate over each snapshot
    for snapshot_name, tags in snapshot_tags.items():
        # Convert MotionTags instance to a list of tag dictionaries
        motion_tags = tags.motion_tags[0]

        # Dictionary to store the integrated tags for each tag type and agent combination
        merged_tags = defaultdict(list)

        # Sort tags by tag type, agents involved, and start time
        sorted_tags = sorted(motion_tags, key=lambda x: (x['tag'], "_".join(sorted(x['agents'])), x['interval'][0]))

        # Integrate tags with small intervals
        for tag in sorted_tags:
            tag_type = tag['tag']
            agents_key = "_".join(sorted(tag['agents']))
            combined_key = (tag_type, agents_key)
            start_time, end_time = tag['interval']

            if not merged_tags[combined_key]:
                # No existing tag, add the new one
                merged_tags[combined_key].append([start_time, end_time])
            else:
                # Check the last added tag to see if it can be merged
                last_tag = merged_tags[combined_key][-1]
                if last_tag[1] + tolerance >= start_time:
                    # Merge the tags by extending the end time
                    last_tag[1] = max(last_tag[1], end_time)
                else:
                    # No merge possible, add as new tag
                    merged_tags[combined_key].append([start_time, end_time])

        # Format the results to match expected output structure
        formatted_tags = []
        for (tag_type, agents_key), intervals in merged_tags.items():
            for interval in intervals:
                formatted_tags.append({
                    'tag': tag_type,
                    'agents': agents_key.split('_'),
                    'interval': tuple(interval),
                    'type': 'unary' if len(agents_key.split('_')) == 1 else 'binary'
                })

        # Store the integrated tags for this snapshot
        integrated_tags[snapshot_name] = MotionTags([formatted_tags])

    return integrated_tags

def remove_short_motion_tags(snapshot_tags, min_duration=20):
    filtered_tags = {}

    # Iterate over each snapshot
    for snapshot_name, tags in snapshot_tags.items():
        # Convert MotionTags instance to a list of tag dictionaries
        motion_tags = tags.motion_tags[0]

        # Filter out short tags
        long_tags = [tag for tag in motion_tags if (tag['interval'][1] - tag['interval'][0]) >= min_duration]

        # Store the filtered tags, update to MotionTags format if necessary
        filtered_tags[snapshot_name] = MotionTags([long_tags])

    return filtered_tags

exclusion_groups = {
    'Accelerate': ['Stopping', 'Decelerate', 'KeepSpeed', 'Parked'],
    'Stopping': ['Accelerate', 'KeepSpeed', 'Parked'],
    'Decelerate': ['Accelerate', 'Stopping', 'Parked'],
    'KeepSpeed': ['Accelerate', 'Stopping', 'Decelerate', 'Parked'],
    'Parked': ['Accelerate', 'Stopping', 'Decelerate', 'KeepSpeed', 'Straight', 'KeepLane'],
    'LeftTurn': ['RightTurn', 'Straight'],
    'RightTurn': ['LeftTurn', 'Straight'],
    'Straight': ['LeftTurn', 'RightTurn', 'Parked'],
    'LeftLaneChange': ['RightLaneChange', 'KeepLane'],
    'RightLaneChange': ['LeftLaneChange', 'KeepLane'],
    'KeepLane': ['LeftLaneChange', 'RightLaneChange', 'Parked'],
}

# Define priorities, lower number means higher priority
priority_dict = {
    'LeftTurn': 1,
    'RightTurn': 1,
    'Straight': 3,
    'LeftLaneChange': 1,
    'RightLaneChange': 1,
    'KeepLane': 3,
    'Accelerate': 1,
    'Stopping': 1,
    'Decelerate': 1,
    'KeepSpeed': 3,
    'Parked': 2
}

def resolve_and_adjust_conflicts(snapshot_tags, exclusion_groups, priority_dict):
    resolved_tags = {}

    for snapshot_name, tags in snapshot_tags.items():
        motion_tags = tags.motion_tags[0]
        sorted_tags = sorted(motion_tags, key=lambda x: x['interval'][0])

        current_tags = []  # Holds the currently active tags
        resolved_list = []  # Final list of resolved tags

        for tag in sorted_tags:
            new_start, new_end = tag['interval']
            new_agents = tag['agents']  # Use list directly
            new_priority = priority_dict.get(tag['tag'], float('inf'))  # Get priority or use a high value if not defined

            # New list for holding adjusted tags
            adjusted_current_tags = []

            for current_tag in current_tags:
                current_start, current_end = current_tag['interval']
                current_agents = current_tag['agents']
                current_priority = priority_dict.get(current_tag['tag'], float('inf'))

                # Check for actual overlap and same agents
                if (new_agents == current_agents and 
                    tag['tag'] in exclusion_groups.get(current_tag['tag'], []) and
                    max(current_start, new_start) < min(current_end, new_end)):  # Check for actual overlap
                    # Compare priorities
                    if current_priority < new_priority:
                        # Current tag has higher priority, adjust new tag
                        new_start = current_end
                    elif new_priority < current_priority:
                        # New tag has higher priority, adjust current tag
                        if current_start < new_start:
                            adjusted_current_tags.append({'tag': current_tag['tag'], 'agents': current_agents, 'interval': (current_start, new_start), 'type': current_tag['type']})
                        current_end = new_start
                    else:
                        # Same priority, keep earlier tag or split based on start times
                        if new_start > current_start:
                            adjusted_current_tags.append({'tag': current_tag['tag'], 'agents': current_agents, 'interval': (current_start, new_start), 'type': current_tag['type']})
                            current_end = new_start

                # Add current tag if not adjusted or eliminated
                if current_start < current_end:
                    adjusted_current_tags.append({'tag': current_tag['tag'], 'agents': current_agents, 'interval': (current_start, current_end), 'type': current_tag['type']})

            # Add the new tag if still valid
            if new_start < new_end:
                adjusted_current_tags.append({'tag': tag['tag'], 'agents': new_agents, 'interval': (new_start, new_end), 'type': tag['type']})

            current_tags = adjusted_current_tags

        # Merge contiguous or identical tags
        if current_tags:
            merged_tags = [current_tags[0]]
            for tag in current_tags[1:]:
                last_tag = merged_tags[-1]
                if (tag['tag'] == last_tag['tag'] and 
                    tag['agents'] == last_tag['agents'] and 
                    tag['interval'][0] <= last_tag['interval'][1]):
                    # Extend the last tag if they are contiguous or overlapping
                    merged_tags[-1]['interval'] = (last_tag['interval'][0], max(last_tag['interval'][1], tag['interval'][1]))
                else:
                    merged_tags.append(tag)
            resolved_list.extend(merged_tags)
        else:
            resolved_list.extend(current_tags)

        # Store the resolved tags in the MotionTags format
        resolved_tags[snapshot_name] = MotionTags([resolved_list])

    return resolved_tags
