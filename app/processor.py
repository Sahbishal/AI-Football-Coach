import cv2
from ultralytics import YOLO
import numpy as np
import json
from sklearn.cluster import KMeans

class VideoProcessor:
    def __init__(self, model_path="yolov8n.pt"):
        # Load a pretrained YOLOv8n model
        self.model = YOLO(model_path)

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        # Clamp to frame
        h, w, _ = frame.shape
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        # Focus on top half (jersey)
        crop = crop[0:crop.shape[0]//2, :]
        if crop.size == 0: return None
        
        # Get average color (B, G, R)
        return np.mean(crop, axis=(0, 1))

    def assign_teams(self, track_colors):
        # track_colors: {track_id: [color1, color2, ...]}
        # Compute average color for each track
        track_avgs = []
        track_ids = []
        
        for tid, colors in track_colors.items():
            track_ids.append(tid)
            track_avgs.append(np.mean(colors, axis=0))
            
        if len(track_avgs) < 2:
            return {tid: 0 for tid in track_ids}
            
        # Cluster into 2 teams
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(track_avgs)
        labels = kmeans.labels_
        
        return {tid: int(label) for tid, label in zip(track_ids, labels)}

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        match_data = []
        track_colors = {} # {track_id: [colors]}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 tracking on the frame
            # persist=True is crucial for tracking across frames
            results = self.model.track(frame, persist=True, classes=[0, 32], verbose=False)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Extract tracking data
            frame_info = {
                "frame": frame_count,
                "objects": []
            }
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                cls = results[0].boxes.cls.cpu().numpy()
                
                for box, track_id, class_id in zip(boxes, ids, cls):
                    # Collect color for team assignment (only for persons)
                    if int(class_id) == 0:
                        color = self.get_player_color(frame, box)
                        if color is not None:
                            if int(track_id) not in track_colors:
                                track_colors[int(track_id)] = []
                            track_colors[int(track_id)].append(color)

                    frame_info["objects"].append({
                        "id": int(track_id),
                        "class": int(class_id), # 0=person, 32=ball
                        "bbox": box.tolist()
                    })
            
            match_data.append(frame_info)
            
            # Write the frame into the file
            out.write(annotated_frame)
            frame_count += 1
            
        cap.release()
        out.release()
        
        # Assign teams
        team_map = self.assign_teams(track_colors)
        
        # Calculate Stats
        team_stats = {0: {"possession": 0, "touches": 0}, 1: {"possession": 0, "touches": 0}, -1: {"possession": 0}}
        events = []
        
        # Update match_data with teams and stats
        for i, frame_info in enumerate(match_data):
            ball = next((obj for obj in frame_info["objects"] if obj["class"] == 32), None)
            players = [obj for obj in frame_info["objects"] if obj["class"] == 0]
            
            # Assign teams to players
            for obj in players:
                if obj["id"] in team_map:
                    obj["team"] = team_map[obj["id"]]
                else:
                    obj["team"] = -1
            
            # Possession logic
            if ball:
                ball_center = np.array([(ball["bbox"][0]+ball["bbox"][2])/2, (ball["bbox"][1]+ball["bbox"][3])/2])
                min_dist = float('inf')
                closest_player = None
                
                for player in players:
                    if player["team"] != -1:
                        p_center = np.array([(player["bbox"][0]+player["bbox"][2])/2, (player["bbox"][1]+player["bbox"][3])/2])
                        dist = np.linalg.norm(ball_center - p_center)
                        if dist < min_dist:
                            min_dist = dist
                            closest_player = player
                
                # If close enough (e.g. 50 pixels), give possession
                if min_dist < 100: # Threshold depends on resolution
                    team_stats[closest_player["team"]]["possession"] += 1
                    team_stats[closest_player["team"]]["touches"] += 1
                    
                    # Simple Event Detection (e.g. "Touch")
                    # To avoid spam, only log if team changed or significant time passed
                    # For now, just store current possession team in frame
                    frame_info["possession_team"] = closest_player["team"]
                else:
                    frame_info["possession_team"] = -1
            else:
                frame_info["possession_team"] = -1

        # Calculate percentages
        total_poss = team_stats[0]["possession"] + team_stats[1]["possession"]
        if total_poss > 0:
            stats_summary = {
                "team_a": {
                    "possession": round(team_stats[0]["possession"] / total_poss * 100, 1),
                    "touches": team_stats[0]["touches"],
                    "suggestion": "Push Left Flank" if team_stats[0]["possession"] > team_stats[1]["possession"] else "Hold Steady"
                },
                "team_b": {
                    "possession": round(team_stats[1]["possession"] / total_poss * 100, 1),
                    "touches": team_stats[1]["touches"],
                    "suggestion": "Counter Attack" if team_stats[1]["possession"] < team_stats[0]["possession"] else "Control Midfield"
                }
            }
        else:
            stats_summary = {"team_a": {"possession": 50}, "team_b": {"possession": 50}}

        return output_path, {
            "metadata": {
                "fps": fps,
                "width": width,
                "height": height
            },
            "stats": stats_summary,
            "frames": match_data
        }
