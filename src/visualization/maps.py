"""
Interactive geospatial map generation for PathSnap.
Supports Folium maps with automatic fallback to native Leaflet.js HTML maps.
Visualizes Raw GPS points, road network geometry, reference route, and predicted route.

Coordinate convention:
- track: [lon, lat, timestamp] -> converted to [lat, lon] for Leaflet/Folium
- predicted_coords / ground_truth_coords: [lat, lon]
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import json
import numpy as np


class StandaloneLeafletMap:
    """Fallback interactive Leaflet map that works without any third-party dependencies."""

    def __init__(self, center_lat: float, center_lon: float, zoom: int = 14):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.layers = []
        self.markers = []
        self.title = "PathSnap Map"

    def add_layer(self, name: str, coords: List[Tuple[float, float]], color: str, weight: float, dash: bool = False):
        self.layers.append({
            "name": name,
            "coords": coords,
            "color": color,
            "weight": weight,
            "dash": dash
        })

    def add_marker(self, lat: float, lon: float, title: str, color: str):
        self.markers.append({"lat": lat, "lon": lon, "title": title, "color": color})

    def to_html(self) -> str:
        layers_json = json.dumps(self.layers)
        markers_json = json.dumps(self.markers)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{self.title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        .legend {{
            background: white; padding: 10px; border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2); font-size: 13px; line-height: 1.5;
        }}
        .legend span {{ display: inline-block; width: 20px; height: 3px; margin-right: 5px; vertical-align: middle; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{self.center_lat}, {self.center_lon}], {self.zoom});
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);

        var layersData = {layers_json};
        var markersData = {markers_json};
        var overlayMaps = {{}};

        layersData.forEach(function(layer) {{
            var poly = L.polyline(layer.coords, {{
                color: layer.color,
                weight: layer.weight,
                dashArray: layer.dash ? '6, 8' : null,
                opacity: 0.85
            }});
            poly.addTo(map);
            overlayMaps[layer.name] = poly;
        }});

        markersData.forEach(function(m) {{
            var marker = L.circleMarker([m.lat, m.lon], {{
                radius: 6,
                fillColor: m.color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.9
            }}).bindPopup('<b>' + m.title + '</b>');
            marker.addTo(map);
        }});

        L.control.layers(null, overlayMaps, {{ collapsed: false }}).addTo(map);

        var legend = L.control({{ position: 'bottomleft' }});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<b>PathSnap Map Legend</b><br>' +
                '<span><span style="background:#2A69AC;"></span> Raw GPS (Dashed)<br>' +
                '<span><span style="background:#2CA02C;"></span> Ground Truth Route<br>' +
                '<span><span style="background:#D62728;"></span> Predicted Route';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>"""

    def save(self, filepath: Union[str, Path]) -> str:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_html())
        return str(p)


def create_folium_map(
    track: np.ndarray,
    predicted_coords: Optional[List[Tuple[float, float]]] = None,
    ground_truth_coords: Optional[List[Tuple[float, float]]] = None,
    road_network_segments: Optional[List[List[Tuple[float, float]]]] = None,
    trip_id: str = "trip",
) -> Any:
    """
    Generate an interactive Folium map or fallback StandaloneLeafletMap.
    
    Layers:
      - Blue Circle Markers & Dashed Line: Raw noisy GPS observations.
      - Gray Lines: Local road-network geometry.
      - Green Polyline: Reference Ground-Truth Route (if available).
      - Red Polyline: Predicted Matched Route from HMM / Classifier.
    """
    center_lat = float(np.mean(track[:, 1]))
    center_lon = float(np.mean(track[:, 0]))

    try:
        import folium

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="CartoDB positron",
        )

        # 1. Road Network Layer (Gray lines)
        if road_network_segments:
            network_group = folium.FeatureGroup(name="Road Network", show=False)
            for seg in road_network_segments[:1500]:
                folium.PolyLine(
                    locations=seg,
                    color="#888888",
                    weight=1.5,
                    opacity=0.6,
                ).add_to(network_group)
            network_group.add_to(m)

        # 2. Raw GPS Observations Layer (Blue markers and dashed line)
        gps_group = folium.FeatureGroup(name="Observed GPS Trajectory", show=True)
        gps_coords = [(float(row[1]), float(row[0])) for row in track]

        folium.PolyLine(
            locations=gps_coords,
            color="#2A69AC",
            weight=2.5,
            opacity=0.7,
            dash_array="5, 10",
            tooltip="Raw GPS Path",
        ).add_to(gps_group)

        folium.Marker(
            location=gps_coords[0],
            popup=f"Start GPS ({trip_id})",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(gps_group)

        folium.Marker(
            location=gps_coords[-1],
            popup=f"End GPS ({trip_id})",
            icon=folium.Icon(color="darkred", icon="stop", prefix="fa"),
        ).add_to(gps_group)

        step = max(1, len(gps_coords) // 50)
        for i in range(0, len(gps_coords), step):
            folium.CircleMarker(
                location=gps_coords[i],
                radius=3,
                color="#1F77B4",
                fill=True,
                fill_color="#1F77B4",
                fill_opacity=0.8,
                tooltip=f"GPS #{i}",
            ).add_to(gps_group)

        gps_group.add_to(m)

        # 3. Reference Route Layer (Green line)
        if ground_truth_coords and len(ground_truth_coords) > 1:
            gt_group = folium.FeatureGroup(name="Ground Truth Route (Reference)", show=True)
            folium.PolyLine(
                locations=ground_truth_coords,
                color="#2CA02C",
                weight=4.0,
                opacity=0.85,
                tooltip="Reference Ground-Truth Route",
            ).add_to(gt_group)
            gt_group.add_to(m)

        # 4. Predicted Matched Route Layer (Red line)
        if predicted_coords and len(predicted_coords) > 1:
            pred_group = folium.FeatureGroup(name="PathSnap Predicted Route", show=True)
            folium.PolyLine(
                locations=predicted_coords,
                color="#D62728",
                weight=3.5,
                opacity=0.9,
                tooltip="Predicted Matched Trajectory",
            ).add_to(pred_group)
            pred_group.add_to(m)

        legend_html = f"""
        <div style="
            position: fixed; 
            bottom: 30px; left: 30px; width: 230px; height: 140px; 
            background-color: white; border:2px solid #ccc; z-index:9999; font-size:12px;
            padding: 10px; border-radius: 6px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
            <b>PathSnap Map Legend ({trip_id})</b><br>
            <span style="color:#2A69AC;">&#9644;&#9644;&#9644;</span> Raw GPS Track (Dashed)<br>
            <span style="color:#2CA02C;">&#9644;&#9644;&#9644;</span> Ground Truth Route<br>
            <span style="color:#D62728;">&#9644;&#9644;&#9644;</span> PathSnap Predicted Route<br>
            <span style="color:#888888;">&#9644;&#9644;&#9644;</span> Road Network Arcs<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    except ImportError:
        # Fallback to StandaloneLeafletMap
        fallback_map = StandaloneLeafletMap(center_lat, center_lon, zoom=14)
        gps_coords = [(float(row[1]), float(row[0])) for row in track]
        fallback_map.add_layer("Observed GPS Trajectory", gps_coords, color="#2A69AC", weight=2.5, dash=True)
        fallback_map.add_marker(gps_coords[0][0], gps_coords[0][1], f"Start GPS ({trip_id})", "#10B981")
        fallback_map.add_marker(gps_coords[-1][0], gps_coords[-1][1], f"End GPS ({trip_id})", "#EF4444")

        if ground_truth_coords and len(ground_truth_coords) > 1:
            fallback_map.add_layer("Ground Truth Route", ground_truth_coords, color="#2CA02C", weight=4.0)

        if predicted_coords and len(predicted_coords) > 1:
            fallback_map.add_layer("Predicted Route", predicted_coords, color="#D62728", weight=3.5)

        return fallback_map


def save_folium_map(m: Any, output_path: Union[str, Path]) -> str:
    """Save folium or fallback leaflet map to HTML file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(p))
    return str(p)
