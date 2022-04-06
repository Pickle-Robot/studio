// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import { Stack, Typography } from "@mui/material";
import { isEmpty } from "lodash";
import { ReactNode, useLayoutEffect, useState } from "react";

import { definitions as commonDefs } from "@foxglove/rosmsg-msgs-common";
import { PanelExtensionContext } from "@foxglove/studio";
import {
  SettingsTree,
  updateSettingsTree,
} from "@foxglove/studio-base/components/SettingsTreeEditor/types";
import ThemeProvider from "@foxglove/studio-base/theme/ThemeProvider";
import { PanelConfig } from "@foxglove/studio-base/types/panels";

import DirectionalPad, { DirectionalPadAction } from "./DirectionalPad";
import { DefaultState } from "./defaultState";

type TeleopPanelProps = {
  context: PanelExtensionContext;
};

type TeleopState = typeof DefaultState;

function ErrorMessage({
  children,
  message,
}: {
  children?: ReactNode;
  message: string;
}): JSX.Element {
  return (
    <Stack
      alignItems="center"
      direction="column"
      spacing={3}
      style={{ maxWidth: "60ch", textAlign: "center" }}
    >
      <Typography variant="h4">{message}</Typography>
      {children}
    </Stack>
  );
}

function TeleopPanel(props: TeleopPanelProps): JSX.Element {
  const { context } = props;
  const { saveState } = context;

  const [currentAction, setCurrentAction] = useState<DirectionalPadAction | undefined>();

  const [panelState, setPanelState] = useState<undefined | TeleopState>(
    (context.initialState as undefined | TeleopState) ?? DefaultState,
  );

  // setup context render handler and render done handling
  const [renderDone, setRenderDone] = useState<() => void>(() => () => {});
  const [colorScheme, setColorScheme] = useState<"dark" | "light">("light");
  useLayoutEffect(() => {
    const interceptor = (previous: PanelConfig, path: string[], value: unknown) => {
      return updateSettingsTree(previous as SettingsTree, path, value);
    };
    context.setSettingsChangeInterceptor(interceptor);

    context.watch("colorScheme");
    context.watch("configuration");
    context.watch("topics");

    context.onRender = (renderState, done) => {
      setRenderDone(() => done);
      if (renderState.colorScheme) {
        setColorScheme(renderState.colorScheme);
      }

      if (isEmpty(renderState.configuration)) {
        saveState(DefaultState);
      } else {
        setPanelState(renderState.configuration as TeleopState);
      }
    };
  }, [context, saveState]);

  const settings = panelState?.settings.tree;

  // advertise topic
  const currentTopic = settings?.fields.topic.value;
  useLayoutEffect(() => {
    if (!currentTopic) {
      return;
    }

    context.advertise?.(currentTopic, "geometry_msgs/Twist", {
      datatypes: new Map([
        ["geometry_msgs/Vector3", commonDefs["geometry_msgs/Vector3"]],
        ["geometry_msgs/Twist", commonDefs["geometry_msgs/Twist"]],
      ]),
    });

    return () => {
      context.unadvertise?.(currentTopic);
    };
  }, [context, currentTopic]);

  useLayoutEffect(() => {
    if (currentAction == undefined || !currentTopic) {
      return;
    }

    const message = {
      linear: {
        x: 0,
        y: 0,
        z: 0,
      },
      angular: {
        x: 0,
        y: 0,
        z: 0,
      },
    };

    function setFieldValue(field: string, value: number) {
      switch (field) {
        case "linear-x":
          message.linear.x = value;
          break;
        case "linear-y":
          message.linear.y = value;
          break;
        case "linear-z":
          message.linear.z = value;
          break;
        case "angular-x":
          message.angular.x = value;
          break;
        case "angular-y":
          message.angular.y = value;
          break;
        case "angular-z":
          message.angular.z = value;
          break;
      }
    }

    switch (currentAction) {
      case DirectionalPadAction.UP:
        setFieldValue(
          settings.children.upButton.fields.field.value,
          settings.children.upButton.fields.value.value,
        );
        break;
      case DirectionalPadAction.DOWN:
        setFieldValue(
          settings.children.downButton.fields.field.value,
          settings.children.downButton.fields.value.value,
        );
        break;
      case DirectionalPadAction.LEFT:
        setFieldValue(
          settings.children.leftButton.fields.field.value,
          settings.children.leftButton.fields.value.value,
        );
        break;
      case DirectionalPadAction.RIGHT:
        setFieldValue(
          settings.children.rightButton.fields.field.value,
          settings.children.rightButton.fields.value.value,
        );
        break;
      case DirectionalPadAction.STOP:
        break;
    }

    // don't publish if rate is 0 or negative - this is a config error on user's part
    if (settings.fields.publishRate.value <= 0) {
      return;
    }

    const intervalMs = (1000 * 1) / settings.fields.publishRate.value;
    context.publish?.(currentTopic, message);
    const intervalHandle = setInterval(() => {
      context.publish?.(currentTopic, message);
    }, intervalMs);

    return () => {
      clearInterval(intervalHandle);
    };
  }, [context, panelState, currentTopic, currentAction, settings]);

  useLayoutEffect(() => {
    renderDone();
  }, [renderDone]);

  const canPublish =
    context.publish != undefined && settings != undefined && settings.fields.publishRate.value > 0;
  const hasTopic = Boolean(currentTopic);
  const enabled = canPublish && hasTopic;

  return (
    <ThemeProvider isDark={colorScheme === "dark"}>
      <Stack height="100%" justifyContent="center" alignItems="center" padding="min(5%, 8px)">
        {!canPublish && (
          <ErrorMessage message="Please connect to a datasource that supports publishing in order to use this panel." />
        )}
        {canPublish && !hasTopic && (
          <ErrorMessage message="Please select a topic in the panel settings in order to use this panel." />
        )}
        {enabled && <DirectionalPad onAction={setCurrentAction} disabled={!enabled} />}
      </Stack>
    </ThemeProvider>
  );
}

export default TeleopPanel;
