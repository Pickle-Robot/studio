// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import ArrowDownIcon from "@mui/icons-material/ArrowDropDown";
import ArrowRightIcon from "@mui/icons-material/ArrowRight";
import LayerIcon from "@mui/icons-material/Layers";
import SettingsIcon from "@mui/icons-material/Settings";
import {
  Collapse,
  Divider,
  ListItem,
  ListItemButton,
  ListItemButtonProps,
  ListItemIcon,
  ListItemProps,
  ListItemText,
  styled as muiStyled,
} from "@mui/material";
import { ChangeEvent, useState } from "react";
import { DeepReadonly } from "ts-essentials";

import { SettingsTreeNode } from "@foxglove/studio";
import Stack from "@foxglove/studio-base/components/Stack";

import { FieldEditor } from "./FieldEditor";
import { VisibilityToggle } from "./VisibilityToggle";

export type NodeEditorProps = {
  defaultOpen?: boolean;
  disableIcon?: boolean;
  divider?: ListItemProps["divider"];
  group?: string;
  icon?: JSX.Element;
  onClick?: ListItemButtonProps["onClick"];
  path: string[];
  secondaryAction?: ListItemProps["secondaryAction"];
  settings?: DeepReadonly<SettingsTreeNode>;
  updateSettings?: (path: string[], value: unknown) => void;
};

const StyledListItem = muiStyled(ListItem, {
  shouldForwardProp: (prop) => prop !== "visible" && prop !== "indent",
})<{
  visible: boolean;
  indent: number;
}>(({ theme, visible, indent = 0 }) => ({
  ".MuiListItemButton-root": {
    paddingLeft: indent === 3 ? theme.spacing(3.5) : theme.spacing(0.5),
    gap: theme.spacing(1),
  },
  ".MuiListItemIcon-root": {
    minWidth: theme.spacing(5),
    opacity: visible ? 0.6 : 0.3,
    display: "flex",
    justifyContent: "flex-end",
  },
  "&:hover": {
    outline: `1px solid ${theme.palette.primary.main}`,
    outlineOffset: -1,

    ".MuiListItemIcon-root": {
      opacity: visible ? 1 : 0.8,
    },
  },
  ...(visible && {
    "@media (pointer: fine)": {
      ".MuiListItemSecondaryAction-root": {
        visibility: "hidden",
      },
      "&:hover": {
        ".MuiListItemSecondaryAction-root": {
          visibility: "visible",
        },
      },
    },
  }),
}));

const LayerOptions = muiStyled("div", {
  shouldForwardProp: (prop) => prop !== "visible" && prop !== "indent",
})<{
  visible: boolean;
  indent: number;
}>(({ theme, visible, indent = 0 }) => ({
  display: "grid",
  gridTemplateColumns: "minmax(128px, 1fr) 200px",
  gridAutoRows: 30,
  padding: theme.spacing(0.5, 1.5, 1),
  paddingLeft:
    (indent === 0 && theme.spacing(7)) ||
    (indent === 1 && theme.spacing(5 * indent + 2)) ||
    (indent === 2 && theme.spacing(2 * indent + 2.5)) ||
    (indent === 3 && theme.spacing(3 * indent + 1)) ||
    theme.spacing(1),
  columnGap: theme.spacing(0.5),
  rowGap: theme.spacing(0.25),
  alignItems: "center",
  opacity: visible ? 1 : 0.6,
}));

export function NodeEditor(props: NodeEditorProps): JSX.Element {
  const {
    icon,
    defaultOpen = true,
    disableIcon = false,
    onClick = () => {},
    settings = {},
    secondaryAction,
    updateSettings = () => {},
  } = props;
  const [open, setOpen] = useState<boolean>(defaultOpen);
  const [visible, setVisiblity] = useState<boolean>(true);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    setVisiblity(event.target.checked);
  };

  const { fields, children } = settings;
  const hasProperties = fields != undefined || children != undefined;

  const fieldEditors = Object.entries(fields ?? {}).map(([key, field]) => (
    <FieldEditor
      key={key}
      field={field}
      update={(value: unknown) => updateSettings([...props.path, key], value)}
    />
  ));

  const childNodes = Object.entries(children ?? {}).map(([key, child]) => (
    <NodeEditor
      disableIcon={props.path.length > 0}
      key={key}
      settings={child}
      updateSettings={updateSettings}
      path={[...props.path, key]}
    />
  ));

  const indent: number = props.path.length;

  return (
    <>
      {(indent > 0 || fieldEditors.length > 0) && (
        <StyledListItem
          indent={indent}
          visible={visible}
          secondaryAction={
            <Stack direction="row" gap={0.5} alignItems="center">
              {secondaryAction}
              <VisibilityToggle edge="end" size="small" checked={visible} onChange={handleChange} />
            </Stack>
          }
          disablePadding
        >
          <ListItemButton
            onClick={(event) => {
              if (hasProperties) {
                setOpen(!open);
              } else {
                onClick(event);
              }
            }}
          >
            <ListItemIcon>
              {hasProperties && <>{open ? <ArrowDownIcon /> : <ArrowRightIcon />}</>}
              {!disableIcon &&
                (icon != undefined ? (
                  icon
                ) : props.path.length > 0 ? (
                  <LayerIcon />
                ) : (
                  <SettingsIcon />
                ))}
            </ListItemIcon>
            <ListItemText
              primary={settings.label ?? "Settings"}
              primaryTypographyProps={{
                noWrap: true,
                variant: "subtitle2",
                color: visible ? "text.primary" : "text.disabled",
              }}
            />
          </ListItemButton>
        </StyledListItem>
      )}
      <Collapse in={open}>
        {fieldEditors.length > 0 && (
          <LayerOptions indent={indent} visible={visible}>
            {fieldEditors}
          </LayerOptions>
        )}
        {indent !== 0 && childNodes}
      </Collapse>
      {indent === 0 && childNodes}
      {indent === 1 && <Divider />}
    </>
  );
}
