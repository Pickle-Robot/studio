// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import {
  ArrowCircleUpOutlined,
  ArrowCircleDownOutlined,
  ArrowCircleRightOutlined,
  ArrowCircleLeftOutlined,
} from "@mui/icons-material";

const geometryMsgOptions = [
  "linear-x",
  "linear-y",
  "linear-z",
  "angular-x",
  "angular-y",
  "angular-z",
];

export const DefaultConfig = {
  format: "settings-tree",
  fields: {
    publishRate: {
      label: "Publish Rate",
      input: "number",
      value: 1,
    },
    topic: {
      label: "Topic",
      input: "string",
      value: "/twist",
    },
  },
  children: {
    upButton: {
      label: "Up Button",
      icon: <ArrowCircleUpOutlined />,
      fields: {
        field: {
          label: "Field",
          input: "select",
          value: "linear-x",
          options: geometryMsgOptions,
        },
        value: { label: "Value", input: "number", value: 1 },
      },
    },
    downButton: {
      label: "Down Button",
      icon: <ArrowCircleDownOutlined />,
      fields: {
        field: {
          label: "Field",
          input: "select",
          value: "linear-x",
          options: geometryMsgOptions,
        },
        value: { label: "Value", input: "number", value: -1 },
      },
    },
    leftButton: {
      label: "Left Button",
      icon: <ArrowCircleLeftOutlined />,
      fields: {
        field: {
          label: "Field",
          input: "select",
          value: "angular-z",
          options: geometryMsgOptions,
        },
        value: { label: "Value", input: "number", value: -1 },
      },
    },
    rightButton: {
      label: "Right Button",
      icon: <ArrowCircleRightOutlined />,
      fields: {
        field: {
          label: "Field",
          input: "select",
          value: "angular-z",
          options: geometryMsgOptions,
        },
        value: { label: "Value", input: "number", value: 1 },
      },
    },
  },
};
