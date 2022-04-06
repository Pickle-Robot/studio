// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import { useCallback } from "react";

import SettingsTreeEditor from "@foxglove/studio-base/components/SettingsTreeEditor";

import { SettingsTree, updateSettingsTree } from "./types";

export default {
  title: "components/SettingsTreeEditor",
  component: SettingsTreeEditor,
};

const DefaultState: SettingsTree = {
  settings: {
    tree: {
      children: {
        background: {
          label: "Background",
          fields: {
            color: { label: "Color", value: "#000000", input: "color" },
          },
        },
        map: {
          label: "Map",
          fields: {
            message_path: {
              label: "Message path",
              input: "string",
              value: "/gps/fix",
            },
            style: {
              label: "Map style",
              value: "Open Street Maps",
              input: "select",
              options: [
                "Open Street Maps",
                "Stadia Maps (Adelaide Smooth Light)",
                "Stadia Maps (Adelaide Smooth Dark)",
                "Custom",
              ],
            },
            api_key: {
              label: "API key (optional)",
              input: "string",
            },
            color_by: {
              label: "Color by",
              value: "Flat",
              input: "toggle",
              options: ["Flat", "Point data"],
            },
            marker_color: {
              label: "Marker color",
              input: "color",
              value: "#ff0000",
            },
          },
        },
        grid: {
          label: "Grid",
          fields: {
            color: {
              label: "Color",
              value: "#248eff",
              input: "color",
            },
            size: {
              label: "Size",
              value: 10,
              input: "number",
            },
            subdivision: {
              label: "Subdivision",
              input: "number",
              value: 9,
            },
          },
        },
        topics: {
          label: "Topics",
          children: {
            lidar_top: {
              label: "/LIDAR_TOP",
              fields: {
                point_size: {
                  label: "Point Size",
                  input: "number",
                  value: 2,
                },
                point_shape: {
                  label: "Point Shape",
                  input: "toggle",
                  value: "Circle",
                  options: ["Circle", "Square"],
                },
                decay_time: {
                  label: "Decay Time (seconds)",
                  input: "number",
                  value: 0,
                },
              },
            },
            lidar_left: {
              label: "/LIDAR_LEFT",
              fields: {
                point_size: {
                  label: "Point Size",
                  input: "number",
                  value: 2,
                },
                point_shape: {
                  label: "Point Shape",
                  input: "toggle",
                  value: "Circle",
                  options: ["Circle", "Square"],
                },
                decay_time: {
                  label: "Decay Time (seconds)",
                  input: "number",
                  value: 0,
                },
              },
            },
            semantic_map: {
              label: "/SEMANTIC_MAP",
              fields: {
                color: {
                  label: "Color",
                  value: "#00ff00",
                  input: "color",
                },
              },
              children: {
                centerline: {
                  label: "centerline",
                  fields: {
                    color: {
                      label: "Color",
                      value: "#00ff00",
                      input: "color",
                    },
                  },
                },
              },
            },
          },
        },
        pose: {
          label: "Pose",
          fields: {
            color: { label: "Color", value: "#ffffff", input: "color" },
            shaft_length: { label: "Shaft length", value: 1.5, input: "number" },
            shaft_width: { label: "Shaft width", value: 1.5, input: "number" },
            head_length: { label: "Head length", value: 2, input: "number" },
            head_width: { label: "Head width", value: 2, input: "number" },
          },
        },
      },
    },
  },
};

export const Default = (): JSX.Element => {
  const [settings, setSettings] = React.useState({ ...DefaultState });

  const updater = useCallback((path: string[], value: unknown) => {
    setSettings((previous) => updateSettingsTree(previous, path, value));
  }, []);

  return (
    <div style={{ overflowY: "auto" }}>
      <SettingsTreeEditor settings={settings} updater={updater} />
    </div>
  );
};
