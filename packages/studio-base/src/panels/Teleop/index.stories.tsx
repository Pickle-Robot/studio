// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import { action } from "@storybook/addon-actions";
import { Story } from "@storybook/react";
import { cloneDeep } from "lodash";

import { PlayerCapabilities } from "@foxglove/studio-base/players/types";
import PanelSetup from "@foxglove/studio-base/stories/PanelSetup";

import { DefaultConfig } from "./config";
import TeleopPanel from "./index";

export default {
  title: "panels/Teleop",
  component: TeleopPanel,
  decorators: [
    (StoryComponent: Story): JSX.Element => {
      return (
        <PanelSetup
          fixture={{ capabilities: [PlayerCapabilities.advertise], publish: action("publish") }}
        >
          <StoryComponent />
        </PanelSetup>
      );
    },
  ],
};

export const Unconfigured = (): JSX.Element => {
  return <TeleopPanel />;
};

export const WithConfig = (): JSX.Element => {
  const config = cloneDeep(DefaultConfig);
  config.fields.topic.value = "chatter";
  return <TeleopPanel overrideConfig={config} />;
};
